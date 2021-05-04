# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import json
import os
import time
from typing import Callable
import tensorflow as tf
import horovod.tensorflow as hvd

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(os.path.dirname(here))  # 05_scaling-DL/
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.parse_args import parse_args_tensorflow as parse_args
import utils.io as io
logger = io.Logger()


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def metric_average(val, name):
    return hvd.allreduce(val, name)


def prepare_datasets(args):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist-%d.npz' % hvd.rank())
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).shuffle(10000).batch(args.test_batch_size)

    return {'training': train_ds, 'testing': test_ds}


@tf.function
def test_step(model, inputs, metrics):
    batch, target = inputs
    predictions = model(batch, training=False)
    metrics['test_accuracy'].update_state(target, predictions)


@tf.function
def train_step(data, model, loss_fn, optimizer, first_batch, compress=True):
    batch, target = data
    with tf.GradientTape() as tape:
        output = model(batch, training=True)
        loss = loss_fn(target, output)

    compression = (
        hvd.Compression.fp16 if compress
        else hvd.Compression.none
    )
    # Horovod: add Horovod Distributed training
    tape = hvd.DistributedGradientTape(tape, compression=compression)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return  loss, output


LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

def train(
    epoch: int,
    dataset: tf.data.Dataset,
    model: tf.keras.models.Model,
    loss_fn: LossFunction,
    optimizer: tf.optimizers.Optimizer,
    args: dict,
    metrics: dict,
):
    #for batch_idx, data in enumerate(dataset):
    train_iter = dataset.take(10000 // hvd.size())
    compress = args.get('fp16_allreduce', True)
    log_interval = args.get('log_interval', 10)
    for batch_idx, data in enumerate(train_iter):
        first_batch = (epoch == 0 and batch_idx == 0)
        loss, output = train_step(data, model, loss_fn, optimizer,
                                  first_batch, compress=compress)
        metrics['train_accuracy'].update_state(data[1], output)

        if batch_idx % log_interval == 0:
            metrics_ = {
                'epoch': epoch,
                'loss': loss,
                'accuracy': metrics['train_accuracy'].result(),
            }
            io.print_metrics(metrics_, pre=f'[{hvd.rank()}] ', logger=logger)


def main(args):
    data = prepare_datasets(args)
    args = args.__dict__

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, [5, 5], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(20, [5, 5], activation='relu'),
        tf.keras.layers.SpatialDropout2D(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = tf.optimizers.Adam(0.001 * hvd.size())

    #  checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    metrics = {
        'train_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
    }

    # Horovod: adjust number of steps based on number of GPUs.
    epoch_times = []
    epochs = args.get('epochs', 10)
    for epoch in range(epochs):
        t0 = time.time()
        train(epoch, data['training'], model, loss_fn, optimizer, args, metrics)

        if epoch > 2:
            epoch_times.append(time.time() - t0)

        if epoch % 5 == 0 and hvd.rank () == 0:
            for batch in data['testing']:
                test_step(model, batch, metrics)

            logger.log(75 * '-')
            logger.log(f'epoch: {epoch}, test_accuracy: {metrics["test_accuracy"].result()}')
            logger.log(75 * '-')

        for _, val in metrics.items():
            val.reset_states()

    if hvd.rank() == 0:
        world_size = hvd.size()
        epoch_times_str = ', '.join(str(x) for x in epoch_times)
        logger.log('Epoch times:')
        logger.log(epoch_times_str)

        outdir = os.path.join(os.getcwd(), 'results_mnist', f'size{world_size}')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        args_file = os.path.join(outdir, f'args_size{world_size}.json')
        logger.log(f'Saving args to: {args_file}.')

        with open(args_file, 'at') as f:
            json.dump(args.__dict__, f, indent=4)

        times_file = os.path.join(outdir,
                                  f'epoch_times_size{world_size}.csv')
        logger.log(f'Saving epoch times to: {times_file}')
        with open(times_file, 'a') as f:
            f.write(epoch_times_str + '\n')

        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting it.
        #  checkpoint_dir = os.path.join(os.getcwd(), 'tf2_mnist_checkpoints')
        #  if not os.path.isdir(checkpoint_dir):
        #      os.makedirs(checkpoint_dir)
        #
        #  logger.log(f'Saving checkpoint to: {checkpoint_dir}')
        #  checkpoint.save(checkpoint_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
