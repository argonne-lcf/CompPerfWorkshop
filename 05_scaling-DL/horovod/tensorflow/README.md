# Tensorflow with Horovod

---

#### Table of Contents

- [Tensorflow with Horovod](#tensorflow-with-horovod)
    + [Running on ThetaKNL](#running-on-thetaknl)
    + [Running on ThetaGPU](#running-on-thetagpu)

---

**Note:** We provide a complete example here: [tf2_hvd_mnist.py](./tf2_hvd_mnist.py).

Below we describe each of the steps necessary to use Horovod for distributed data-parallel training using `TensorFlow >= 2.`

- **Goal:** 
  1. Understand how Horovod works with TensorFlow
  2. Be able to modify existing code to be compatible with Horovod

---

1. **Initialize Horovod**

   After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`. Besides, one can also call `hvd.local_rank()` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank.

   ```python
   import horovod.tensorflow as hvd
   hvd.init()
   ```

2. **Assign GPUs to each rank**

   In this case, we set one GPU per process: ID=`hvd.local_rank()`

   ```python
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
       tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
   ```

3. **Scale the learning rate by the number of workers**

   ```python
   # Horovod: adjust learning rate based on number of GPUs
   optimizer = tf.optimizers.Adam(0.001 * hvd.size())
   ```

4. Decorate our `train_step` function with the `@tf.function` decorator:

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
       
       # Horovod: broadcast initial variable states from rank 0 to all other
       # processes. This is necessary to ensure consistent initialization
       # of all workers when training is started with random weights or
       # restored from a checkpoint.
       #
       # Note: broadcast should be done after the first gradient step
       # to ensure optimizer initialization.
       if first_batch:
           hvd.broadcast_variables(model.variables, root_rank=0)
           hvd.broadcast_variables(optimizer.variables(), root_rank=0)
       
       return loss, output
   ```

5. **Checkpointing _only_ on root rank**

   It is important to let _only_ one process deal with the checkpointing file I/O to prevent a race condition.

   ```python
   if hvd.rank() == 0:
       checkpoint.save(checkpoint_dir)
   ```

6. **Loading data according to rank ID**

   In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different parts of the dataset, and that together they cover the entire dataset at each epoch.

   In general, one has two ways of dealing with the data loading:

   1. Each worker randomly selects one batch of data from the dataset at each step. In this case, each worker can see the entire dataset. It is important to make sure that the different workers have different random seeds so that they will get different data at each step.
   2. Each worker accesses a subset of the dataset. One can manually partition the entire dataset into different partitions and let each rank access one of the distinct partitions.

7. **Adjusting the number of steps per epoch**

   The total number of steps per epoch is `nsamples / hvd.size()`.

### Running on ThetaKNL

Examples demonstrating how to run Horovod on ThetaKNL are available here [ALCF: Simulation, Data, and Learning Workshop for AI: 01--Distributed Deep Learning](https://github.com/argonne-lcf/sdl_ai_workshop/01_distributedDeepLearning/README.md).

### Running on ThetaGPU

1. Login to Theta:

   ```bash
   # to theta login node from your local machine
   ssh username@theta.alcf.anl.gov
   ```

2. Login to ThetaGPU service node (this is where we can submit jobs directly to ThetaGPU):

   ```bash
   # to thetaGPU service node (sn) from theta login node
   ssh username@thetagpusn1
   ```

3. Submit an interactive job to ThetaGPU

   ```bash
   # should be ran from a service node, thetagpusn1
   qsub -I -A Comp_Perf_Workshop -n 1 -t 00:30:00 -O ddp_tutorial --attrs=pubnet=true
   ```

4. Once your job has started, load the `conda/tensorflow` module and activate the base conda environment

   ```bash
   module load conda/tensorflow
   conda activate base
   ```

5. Clone the `CompPerfWorkshop-2021` github repo (if you haven't already):

   ```bash
   git clone https://github.com/argonne-lcf/CompPerfWorkshop-2021
   ```

6. Navigate into the `horovod/tensorflow/` directory and run the example:

   ```bash
   cd CompPerfWorkshop-2021/05_scaling-DL/horovod/tensorflow
   mpirun -np 8 --verbose python3 ./tf2_hvd_mnist.py --batch_size=256 --epochs=10 > training.log&
   ```