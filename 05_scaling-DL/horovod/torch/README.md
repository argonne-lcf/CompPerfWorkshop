# PyTorch with Horovod

**Note:** We provide an example script, available here: [./horovod/torch/torch_cifar10_hvd.py](./torch/torch_cifar10_hvd.py)

Using Horovod + PyTorch is similar to the procedure described above for TensorFlow.

1. **Initialize Horovod**

   After this initialization, the rank ID and the number of processes can be referred to as `hvd.rank()` and `hvd.size()`. Additionally, we can also call `hvd.local_rank()` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank.

   ```python
   import horovod.torch as hvd
   hvd.init()
   ```

2. **Assign GPUs to each rank**

   In this case, we set one GPU per process: ID = `hvd.local_rank()`

   ```python
   torch.cuda.set_device(hvd.local_rank())
   ```

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

## Results

**PyTorch** (time for 32 epochs)

| GPUs | cifar10 (s) | MNIST (s) |
| :--: | :---------: | :-------: |
|  1   |    522.3    |   499.8   |
|  2   |    318.8    |   283.9   |
|  4   |    121.4    |   100.4   |
|  8   |    73.5     |   58.8    |
|  16  |    79.1     |   63.8    |
|  32  |    81.1     |   55.7    |

<img src="../../images/pytorch_scaling_ddp.png" alt="torch_thetaGPU" style="zoom: 33%;" />

