# Distributed Training with [Horovod][1]

---

#### Table of Contents

- [Distributed Training with Horovod](#distributed-training-with-horovod1)
  * [Model Parallelism and Data Parallelism](#model-parallelism-and-data-parallelism)
    + [Example](#example)
  * [Horovod Data Parallel Frameworks](#horovod-data-parallel-frameworks3)
    + [Tensorflow with Horovod](#tensorflow-with-horovod)
  * [PyTorch with Horovod](#pytorch-with-horovod)
  * [Handson](#handson)
  * [Additional References](#additional-references)

---

**Author**: Sam Foreman ([foremans@anl.gov](mailto:foremans@anl.gov))

**Note**:  Adapted from original material [here](https://github.com/argonne-lcf/sdl_ai_workshop/blob/master/01_distributedDeepLearning/Horovod/README.md), written by __[Huihuo Zheng](mailto:huihuo.zheng@anl.gov)__ and __[Corey Adams](mailto:corey.adams@anl.gov)__.

**Goal:**

1. Understand how to run jobs on Theta / ThetaGPU

2. Get familiar with the software frameworks on Theta / ThetaGPU

3. Understand Data Parallelism (scaling efficiency, warmup, etc)

4. Know how to modify your code with Horovod

---

## Model Parallelism and Data Parallelism

1. **Model parallelization:** In this scheme, disjoint subsets of a neural network are assigned to different devices. Therefore, all the computation associated with the subsets are distributed. Communication happens between devices whenever there is dataflow between two subsets. Model parallelization is suitable when the model is too large to fit into a single device (CPU/GPU) because of the memory capacity. However, partitionining the model into different subsets is not an easy task, and there might potentially introduce load imbalance issues limiting the scaling efficiency.
2. **Data parallelization:** In this scheme, all of the workers own a replica of the model. The global batch of data is split into multiple minibatches and processed by different workers. Each worker computes the corresponding loss and gradients with respect to the data it possesses. Before the updating of the parameters at each epoch, the loss and gradients are averaged among all the workers through a collective operation. This scheme is relatively simple to implement. `MPI_Allreduce` is the only communication operation required.
   1. Our recent presentation about the data parallel training can be found here: https://youtu.be/930yrXjNkgM

### Example:

- How the model **weights** are split over cores ([image credit][2]):

  - Shapes of different sizes in this row represent larger weight matrics in the networks' layers.

  ![weights](../assets/weights.png)

- How the **data** is split over cores:

  ![data](../assets/data.png)

<!---![distributed](../assets/distributed.png)--->

## [Horovod Data Parallel Frameworks][3]

![Horovod](../assets/horovod.png)

[1]: https://github.com/horovod/horovod
[2]: https://venturebeat.com/2021/01/12/google-trained-a-trillion-parameter-ai-language-model/
[3]: https://horovod.readthedocs.io/en/stable/

### Tensorflow with Horovod

**Note:** We provide an example script, available here: [./horovod/tf2hvd_mnist.py](./horovod/tf2hvd_mnist.py).

1. **Initialize Horovod**

   ```python
   import horovod.tensorflow as hvd
   hvd.init()
   ```

   After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`. Besides, one can also call `hvd.local_rank()` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank.

2. **Assign GPUs to each rank**

   ```python
   # Get the list of GPUs
   gpus = tf.config.experimental.list_physical_devices('GPU')
   # Pin GPU to the rank
   tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
   ```

   In this case, we set one GPU per process: ID=`hvd.local_rank()`

3. **Scale the learning rate by the number of workers**

   ```python
   opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
   ```

   Typically, since we use multiple workers, if we keep the local `batch_size` on each rank the same, the global batch size increases by $n$ times (where $n$ is the number of workers). The learning rate should increase proportionally (assuming that the initial learning rate is 0.01).

4. **Wrap** `tf.GradientTape` **with** `hvd.DistributedGradientTape`

   ```python
   tape = hvd.DistributedGradientTape(tape)
   ```

   This ensures that our gradients will be averaged across workers when back propagating.

5. **Broadcast the model from rank 0**

   ```python
   hvd.broadcast_variables(model.variables, root_rank=0)
   hvd.broadcast_variables(opt.variables(), root_rank=0)
   ```

   This ensures that all workers start from the same initial point.

6. **Checkpointing _only_ on root rank**

   ```python
   if hvd.rank() == 0:
       checkpoint.save(checkpoint_dir)
   ```

   It is important to let _only_ one process deal with the checkpointing file I/O to prevent a race condition.

7. **Loading data according to rank ID**

   In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different parts of the dataset, and that together they cover the entire dataset at each epoch.

   In general, one has two ways of dealing with the data loading:

   1. Each worker randomly selects one batch of data from the dataset at each step. In this case, each worker can see the entire dataset. It is important to make sure that the different workers have different random seeds so that they will get different data at each step.
   2. Each worker accesses a subset of the dataset. One can manually partition the entire dataset into different partitions and let each rank access one of the distinct partitions.

8. **Adjusting the number of steps per epoch**

   The total number of steps per epoch is `nsamples / hvd.size()`.

## PyTorch with Horovod

Using Horovod + PyTorch is similar to the procedure described above for TensorFlow.

Below, we omit the explanation for those steps which are logically equivalent (but syntatically different) to the TensorFlow example above, and simply provide the necessary code snippet to accomplish each item.

1. **Initialize Horovod**

   ```python
   import horovod.torch as hvd
   hvd.init()
   ```

   After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`. Additionally, we can also call `hvd.local_rank()` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank.

2. **Assign GPUs to each rank**

   ```python
   torch.cuda.set_device(hvd.local_rank())
   ```

   In this case, we set one GPU per process: ID = `hvd.local_rank()`

3. **Scale the learning rate**

   If we are using $n$ workers, the global batch usually increases $n$ times. The learning rate should increase proportionally as follows (assuming intial learning rate is `0.01`).

   ```python
   from torch import optim
   optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)
   ```

4. **Wrap the optimizer with `hvd.DistributedOptimizer`**

   ```python
   optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression)
   ```

5. **Broadcast the model from rank 0**

   This ensures all workers start from the same point.

   ```python
   hvd.broadcast_parameters(model.state_dict(), root_rank=0)
   hvd.broadcast_optimizer_state(optimizer, root_rank=0)
   ```

6. **Loading data according to the rank ID**

   One minor difference from the TensorFlow example is that PyTorch has some internal functions for dealing with parallel distribution of data

   ```python
   transform = transforms.Compose([
       transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
   ])
   train_dataset = datasets.MNIST('datasets/', train=True, download=True, transform=transform)
   train_sampler = torch.utils.data.distributed.DistributedSampler(
       train_dataset,
       num_replicas=hvd.size(),
       rank=hvd.rank()
   )
   train_loader = torch.utils.data.DataLoader(
   	train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
   )
   ```

   For simplicity, we define a `DataObject` for grouping a datasets' data, sampler, and loader objects using a `dataclass`:

   ```python
   from dataclasses import dataclass
   
   @dataclass
   class DataObject:
       dataset: torch.utils.data.Dataset    # accessible via `DataObject.dataset`, etc.
       sampler: torch.utils.data.Sampler
       loader: torch.utils.data.DataLoader
   ```

   In both cases, the total number of steps per epoch is `nsamples / hvd.size()`.

7. **Checkpointing _only_ from root rank**

   It is important to only let one process be responsible for checkpointing I/O to prevent race conditions which might jeopardize the integrity of the checkpoint.

   ```python
   if hvd.rank() == 0:
       checkpoint.save(checkpoint_dir)
   ```

8. **Average metrics across all workers**

   Notice that in the distributed training, any tensors are local to each worker. In order to get the global averaged value, we can use `hvd.allreduce`. Below we provide an example

   ```python
   def tensor_average(val, name):
       tensor = torch.tensor(val)
       avg_tensor = hvd.allreduce(tensor, name=name) if WITH_HVD else tensor
       return avg_tensor.item()
   ```

We provide an example in [`./horovod/torch/torch_mnist_hvd.py`](./horovod/torch/torch_mnist_hvd.py)

## Handson

- [`./thetagpu.md`](./thetagpu.md)
- [`./theta.md`](./theta.md)

## Additional References

1. Sergeev, A., Del Balso, M. (2017) Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow. Retrieved from https://eng.uber.com/horovod/

2. Sergeev, A. (2017) Horovod - Distributed TensorFlow Made Easy. Retrieved from https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy

3. Sergeev, A., Del Balso, M. (2018) Horovod: fast and easy distributed deep learning in TensorFlow. Retrieved from arXiv:**1802.05799**