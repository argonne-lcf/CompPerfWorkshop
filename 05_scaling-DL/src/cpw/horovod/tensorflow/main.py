"""
horovod/tensorflow/main.py

Simple example demonstrating how to use Horovod with Tensorflow for data
parallel distributed training.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os

import time
from typing import Optional


import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import hydra

from omegaconf import DictConfig
from pathlib import Path

log = logging.getLogger(__name__)
tf.autograph.set_verbosity(0)


hvd.init()
RANK = hvd.rank()
SIZE = hvd.size()
LOCAL_RANK = hvd.local_rank()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(
        gpus[hvd.local_rank()],
        'GPU'
    )


Tensor = tf.Tensor
Model = tf.keras.models.Model
TF_FLOAT = tf.keras.backend.floatx()


def metric_average(x: Tensor) -> Tensor:
    return x if SIZE == 1 else hvd.allreduce(x, average=True)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.rank = RANK
        self.datasets = self.setup_data()
        self.model = self.build_model()
        self.optimizer = tf.optimizers.Adam(cfg.lr_init * SIZE)
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy()

        self.ckpt_dir = Path(os.getcwd()).joinpath('checkpoints').as_posix()
        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                  max_to_keep=5,
                                                  directory=self.ckpt_dir)
        self.ntest = self.ntest_samples // SIZE // cfg.batch_size
        self.ntrain = self.ntrain_samples // SIZE // cfg.batch_size

    def save_checkpoint(self):
        self.manager.save()

    def setup_data(self) -> dict[str, tf.data.Dataset]:
        (xtrain, ytrain), (xtest, ytest) = (
            tf.keras.datasets.mnist.load_data(
                Path(self.cfg.data_dir).joinpath(
                    f'mnist-{hvd.rank()}.npz'
                ).as_posix()
            )
        )

        train_dset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(xtrain[..., tf.newaxis] / 255.0, TF_FLOAT),
             tf.cast(ytrain, tf.int64))
        )
        test_dset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(xtest[..., tf.newaxis] / 255.0, TF_FLOAT),
             tf.cast(ytest, tf.int64))
        )
        self.ntrain_samples = len(list(train_dset))
        self.ntest_samples = len(list(test_dset))
        train_dset = train_dset.repeat().shuffle(self.cfg.buffer_size)
        test_dset = test_dset.shard(
            num_shards=SIZE,
            index=RANK,
        ).repeat()

        return {
            'train': train_dset.batch(self.cfg.batch_size),
            'test': test_dset.batch(self.cfg.batch_size),
        }

    def build_model(self) -> Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        return model

    @tf.function
    def train_step(
            self,
            data: Tensor,
            target: Tensor,
            first_batch: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor]:
        with tf.GradientTape() as tape:
            probs = self.model(data, training=True)
            loss = self.loss_fn(target, probs)
            pred = tf.math.argmax(probs, axis=1)
            acc = tf.math.reduce_sum(
                tf.cast(tf.math.equal(pred, target), TF_FLOAT)
            )

        # Horovod: add Horovod DistributedGradientTape
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        updates = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(updates)
        # Horovod: broadcast initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or restored
        # from a checkpoint

        # NOTE: Broadcast should be done after the first gradient step to
        # ensure optimizer initialization
        if first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return loss, acc

    def train_epoch(
            self,
            epoch: Optional[int] = None
    ) -> dict[str, Tensor]:
        epoch = 0 if epoch is None else epoch
        nstep = self.ntrain_samples // SIZE // self.cfg.batch_size
        batch = self.datasets['train'].take(nstep)
        t0 = time.time()
        metrics = {}
        training_acc = 0.0
        running_loss = 0.0
        for bidx, (data, target) in enumerate(batch):
            loss, acc = self.train_step(
                data, target,  # type:ignore
                first_batch=(bidx == 0 and epoch == 0),
            )
            training_acc += acc
            running_loss += loss.numpy()
            if RANK == 0 and bidx % self.cfg.logfreq == 0:
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - t0,
                    'running_loss': running_loss / (self.ntrain_samples // SIZE),
                    'batch_loss': loss / self.cfg.batch_size,
                    'acc': training_acc / (self.ntrain_samples // SIZE),
                    'batch_acc': acc / self.cfg.batch_size,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{self.ntrain_samples // SIZE}'
                        f' ({100. * bidx / nstep:.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        running_loss = running_loss / self.ntrain_samples
        training_acc = training_acc / self.ntrain_samples
        training_acc = metric_average(training_acc)
        loss_avg = metric_average(running_loss)

        return {'loss': loss_avg, 'acc': training_acc}

    def test(self) -> Tensor:
        test_batch = self.datasets['test'].take(self.ntest)
        total = 0
        correct = 0

        for data, target in test_batch:
            probs = self.model(data, training=False)
            pred = tf.math.argmax(probs, axis=1)
            total += target.shape[0]
            correct += tf.reduce_sum(
                tf.cast(tf.math.equal(pred, target), TF_FLOAT)
            )
        return correct / tf.constant(total, dtype=TF_FLOAT)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    epoch_times = []
    start = time.time()
    trainer = Trainer(cfg)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        metrics = trainer.train_epoch(epoch)
        epoch_times.append(time.time() - t0)

        if epoch % cfg.logfreq == 0 and RANK == 0:
            acc = trainer.test()
            astr = f'[TEST] Accuracy: {acc:.0f}%'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={metrics["loss"]:.4f}',
                f'acc={metrics["acc"] * tf.constant(100., TF_FLOAT):.0f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

            trainer.save_checkpoint()

    log.info(f'Total training time: {time.time() - start} seconds')
    log.info(
        f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    )


if __name__ == '__main__':
    main()
