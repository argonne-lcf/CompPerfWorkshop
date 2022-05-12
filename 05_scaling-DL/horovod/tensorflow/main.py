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

import horovod.tensorflow as hvd
import numpy as np
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from pathlib import Path

log = logging.getLogger(__name__)


hvd.init()
RANK = hvd.rank()
SIZE = hvd.size()
LOCAL_RANK = hvd.local_rank()


Tensor = tf.Tensor
Model = tf.keras.models.Model
TF_FLOAT = tf.keras.backend.floatx()
BUFFER_SIZE = 10000


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[LOCAL_RANK], 'GPU')


def metric_average(x: Tensor) -> Tensor:
    return x if SIZE == 1 else hvd.allreduce(x, average=True)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.rank = RANK
        self.datasets = self.setup_data()
        # self.ntest_samples = len(list(dsets['test']))
        # self.ntrain_samples = len(list(dsets['train']))
        self.ntest = self.ntest_samples // SIZE // cfg.batch_size
        self.ntrain = self.ntrain_samples // SIZE // cfg.batch_size
        # shuffle the dataset, with a shuffle buffer set to be 1000
        # train_dset = dsets['train'].repeat().shuffle(BUFFER_SIZE)
        # test_dset = dsets['test'].shard(num_shards=SIZE, index=RANK).repeat()
        self.model = self.build_model()
        self.optimizer = tf.optimizers.Adam(cfg.lr_init * SIZE)
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy()
        self.ckpt_dir = Path(os.getcwd()).joinpath('checkpoints').as_posix()
        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.optimizer)

    def save_checkpoint(self):
        self.checkpoint.save(self.ckpt_dir)

    def setup_data(self) -> dict[str, tf.data.Dataset]:
        (xtrain, ytrain), (xtest, ytest) = (
            tf.keras.datasets.mnist.load_data(
                Path(self.cfg.data_dir).joinpath(
                    f'mnist-{hvd.rank()}.npz'
                ).as_posix()
            )
        )
        # ntrain_samples = len(xtrain)
        # global_batch_size = SIZE * self.cfg.batch_size

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
        train_dset = train_dset.repeat().shuffle(BUFFER_SIZE)
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
            tf.keras.layers.Dropout(0.25),
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
            correct = tf.math.equal(pred, target)
            # eq = tf.math.equal(pred, target)
            acc = tf.math.reduce_mean(tf.cast(correct, TF_FLOAT))
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
    ) -> tuple[Tensor, Tensor]:
        epoch = 0 if epoch is None else epoch
        running_acc = tf.constant(0.0)
        running_loss = tf.constant(0.0)
        nstep = self.ntrain_samples // SIZE // self.cfg.batch_size
        batch = self.datasets['train'].take(nstep)
        t0 = time.time()
        for bidx, (data, target) in enumerate(batch):
            loss, acc = self.train_step(
                data, target,  # type:ignore
                first_batch=(bidx == 0 and epoch == 0),
            )
            running_acc += acc
            running_loss += loss
            if RANK == 0 and bidx % self.cfg.logfreq == 0:
                log.info(' '.join([
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{self.ntrain_samples // SIZE}'
                        f' ({100. * bidx / nstep:.0f}%)]'
                    ),
                    f'dt={time.time() - t0:.4f}',
                    f'loss={loss:.6f}',
                    f'acc={acc:.6f}',
                ]))

        running_acc = metric_average(running_acc / nstep)
        running_loss = metric_average(running_loss / nstep)
        if RANK == 0:
            summary = '  '.join([
                '[TRAIN]',
                f'dt={time.time() - t0:.6f}',
                f'loss={running_loss:.4f}',
                f'acc={running_acc * tf.constant(100.0):.2f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        return running_loss, running_acc

    @tf.function
    def test_step(self, data, target):
        probs = self.model(data, training=False)
        pred = tf.math.argmax(probs, axis=1)
        eq = tf.math.equal(pred, target)
        acc = tf.math.reduce_mean(tf.cast(eq, TF_FLOAT))
        loss = self.loss_fn(target, probs)

        return loss, acc

    def test(self) -> tuple[Tensor, Tensor]:
        test_acc = tf.constant(0.)
        test_loss = tf.constant(0.)
        test_batch = self.datasets['test'].take(self.ntest)
        t0 = time.time()
        for _, (data, target) in enumerate(test_batch):
            loss, acc = self.test_step(data, target)  # type:ignore
            test_acc += acc
            test_loss += loss

        test_loss = metric_average(test_loss / self.ntest)
        test_acc = metric_average(test_acc / self.ntest)
        dt_test = time.time() - t0
        if RANK == 0:
            summary = ', '.join([
                '[TEST]',
                f'dt={dt_test:.6f}',
                f'loss={(test_loss):.4f}',
                f'acc={(test_acc * tf.constant(100.0)):.2f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        return test_loss, test_acc


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    start = time.time()

    epoch_times = []
    trainer = Trainer(cfg)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        _ = trainer.train_epoch(epoch)
        _ = trainer.test()
        epoch_times.append(time.time() - t0)
        if RANK == 0:
            trainer.save_checkpoint()

    log.info(f'Total training time: {time.time() - start} seconds')
    log.info(
        f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    )


if __name__ == '__main__':
    main()
