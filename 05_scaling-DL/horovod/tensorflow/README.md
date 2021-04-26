# Tensorflow with Horovod

**Note:** We provide an example script, available here: [./horovod/tf2hvd_mnist.py](./horovod/tf2hvd_mnist.py).

1. **Initialize Horovod**

   After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`. Besides, one can also call `hvd.local_rank()` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank.

   ```python
   import horovod.tensorflow as hvd
   hvd.init()
   ```

2. **Assign GPUs to each rank**

   In this case, we set one GPU per process: ID=`hvd.local_rank()`

   ```python
   # Get the list of GPUs
   gpus = tf.config.experimental.list_physical_devices('GPU')
   # Pin GPU to the rank
   tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
   ```

3. **Scale the learning rate by the number of workers**

   In this case, we set one GPU per process: ID=`hvd.local_rank()`

   ```python
   opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
   ```

4. **Wrap** `tf.GradientTape` **with** `hvd.DistributedGradientTape`

   This ensures that our gradients will be averaged across workers when back propagating.

   ```python
   tape = hvd.DistributedGradientTape(tape)
   ```

5. **Broadcast the model from rank 0**

   This ensures that all workers start from the same initial point.

   ```python
   hvd.broadcast_variables(model.variables, root_rank=0)
   hvd.broadcast_variables(opt.variables(), root_rank=0)
   ```

6. **Checkpointing _only_ on root rank**

   It is important to let _only_ one process deal with the checkpointing file I/O to prevent a race condition.

   ```python
   if hvd.rank() == 0:
       checkpoint.save(checkpoint_dir)
   ```

7. **Loading data according to rank ID**

   In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different parts of the dataset, and that together they cover the entire dataset at each epoch.

   In general, one has two ways of dealing with the data loading:

   1. Each worker randomly selects one batch of data from the dataset at each step. In this case, each worker can see the entire dataset. It is important to make sure that the different workers have different random seeds so that they will get different data at each step.
   2. Each worker accesses a subset of the dataset. One can manually partition the entire dataset into different partitions and let each rank access one of the distinct partitions.

8. **Adjusting the number of steps per epoch**

   The total number of steps per epoch is `nsamples / hvd.size()`.