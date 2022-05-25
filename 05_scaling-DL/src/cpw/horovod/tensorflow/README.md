# Horovod with Tensorflow

Authors: Sam Foreman [foremans@anl.gov](mailto:///foremans@anl.gov), Huihuo Zheng [huihuo.zheng@anl.gov](mailto:///huihuo.zheng@anl.gov)

**Note:** We provide a complete example in [`./main.py`](./main.py)

Below we describe each of the steps necessary to use Horovod for distributed data-parallel training using `tensorflow >= 2.x`

Horovod core principles are based on [MPI](http://mpi-forum.org/) concepts such as _size_, _rank_, _local rank_, **allreduce**, **allgather**, **broadcast**, and **alltoall**. See [this page](https://github.com/horovod/horovod/blob/master/docs/concepts.rst) for more details.[^1]

**Goal:**
1. Understand how Horovod works with TensorFlow
2. Be able to modify existing code to be compatible with Horovod

**Steps to use Horovod:**
1. [Initialize Horovod](#Initialize%20Horovod)
2. [Assign GPUs to each rank](#Assign%20GPUs%20to%20each%20rank)
3. [Scale the learning rate](#Scale%20the%20learning%20rate)
4. [Distribute Gradients and Broadcast Variables](#Distribute%20Gradients%20and%20Broadcast%20Variables)
5. [Deal with Data](#Deal%20with%20Data)
6. [Average across workers](#Average%20across%20workers)
7. [Checkpoint only on root rank](#Checkpoint%20only%20on%20root%20rank)

---
## Initialize Horovod
After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`, whereas `hvd.local_rank()` refers to the local rank ID within a node.

This is useful when we are trying to assign GPUs to each rank
```python
import horovod as hvd
hvd.init()
```
  
  ---
## Assign GPUs to each rank
In this case, we set one GPU per process ID `hvd.local_rank()`
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      local_rank = hvd.local_rank()
      tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')
```

---
## Scale the learning rate
We scale the learning rate by the number of workers to account for the increased batch size.

This is trivial in tensorflow via:
```python
# Horovod: adjust learning rate based on number of GPUs
optimizer = tf.optimizers.Adam(lr_init * hvd.size())
```

---  
## Distribute Gradients and Broadcast Variables
To use `tensorflow` for distributed training with Horovod:
1. At the start of training we must make sure that all of the workers are initialized consistently by broadcasting our model and optimizer states from the chief (`rank = 0`) worker
2. Wrap our optimizer with the `hvd.DistributedOptimizer`

Explicitly,
```python
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
  # Wrap `tf.GradientTape` with `hvd.DistributedGradientTape`
  tape = hvd.DistributedGradientTape(tape, compression=compression)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  # Horovod: Broadcast initial variable states from rank 0 to all other
  # processes. This is necessary to ensure consistent initialization
  # of all workers when training is started with random weights or
  # restored from a checkpoint
  #
  # Note: broadcast should be done after the first gradient step
  # to ensure consistent optimizer initialization
  if first_batch:
      hvd.broadcast_variables(model.variables, root_rank=0)
      hvd.broadcast_variables(optimizer.variables, root_rank=0)

  return loss, output
```

---
## Deal with Data
At each training step, we want to ensure that each worker receives unique data.

Naively, this can be done in one of two ways:
1. From each worker, randomly select a minibatch (i.e. each worker can see the *full dataset*). 
    > [!warning] Dont forget your seed!
    >  In this case, it is important that each worker uses different seeds to ensure that they receive unique data
2. Manually partition the data (ahead of time) and assign different sections to different workers (i.e. each worker can only see *their local portion* of the dataset).

```python
TF_FLOAT = tf.keras.backend.get_floatx()

(images, labels), (xtest, ytest) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(images[..., None] / 255.0, TF_FLOAT),
     tf.cast(labels, tf.int64))
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(xtest[..., None] / 255.0, TF_FLOAT),
     tf.cast(ytest, tf.int64))
)

nsamples = len(list(dataset))
ntest = len(list(test_dataset))
dataset = dataset.repeat().shuffle(1000).batch(args.batch_size)
test_dataset = test_dataset.shard(num_shards=hvd.size(), index=hvd.rank()).repeat().batch(args.batch_size)
    
```

---
## Average across workers
Typically we will want to take the global average of the loss across all our workers, for example

```python
global_loss = hvd.allreduce(loss, average=True)
global_acc = hvd.allreduce(acc, average=True)
...
```

---
## Checkpoint only on root rank
It is important to let _only_ one process deal with the checkpointing file I/O to prevent a race condition

```python
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
```

The steps for using Horovod with PyTorch is similar, and are explained in the next section, [Horovod with PyTorch](https://github.com/argonne-lcf/CompPerfWorkshop/blob/main/05_scaling-DL/src/cpw/horovod/torch/README.md)


[^1]: `fas:Github` [`horovod/horovod` Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.](https://github.com/horovod/horovod)
