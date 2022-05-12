# Horovod with PyTorch
The general procedure for using Horovod with `pytorch` is nearly identical to using `tensorflow`. We provide below the necessary steps

**Steps to use Horovod:**
1. [Initialize Horovod](#Initialize%20Horovod)
2. [Assign GPUs to each rank](#Assign%20GPUs%20to%20each%20rank)
3. [Scale the learning rate](#Scale%20the%20learning%20rate)
4. [Distribute Gradients and Broadcast Variables](#Distribute%20Gradients%20and%20Broadcast%20Variables)
5. [Deal with Data](#Deal%20with%20Data)
6. [Global Averaging](#Global%20Averaging)
7. [Checkpointing from rank 0](#Checkpointing%20from%20rank%200)

---
## Initialize Horovod
```python
import horovod.torch as hvd
hvd.init()
```

- Rank ID can be obtained via `hvd.rank()`
- Number of processes can be obtained via `hvd.size()`
- Local rank ID within a node via `hvd.local_rank()`

---
## Assign GPUs to each rank
```python
torch.cuda.set_devices(hvd.local_rank())
```

---
## Scale the learning rate
Again we scale the initial learning rate value by the number of workers (to account for the increased batch size $\propto N_{\mathrm{workers}}$,
```python
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)
```

---
## Distribute Gradients and Broadcast Variables
We can wrap the optimizer using `hvd.DistributedOptimizer` via
```python
optimizer = hvd.DistributedOptimizer(optimizer,
                                     compression=compression,
                                     named_parameters=model.named_parameters())
```

and broadcast the model from `rank = 0`:

```python
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

---
## Deal with Data
```python
train_dataset = datasets.MNIST(
    'datasets/',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.toTensor(),
        transforms.Normalize((.1307,), (0.3081,))
    ])
)

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    **kwargs
)
```

---
## Global Averaging
Average metrics across all workers.

Notice that in distributed training, all tensors are local to each worker. In order to get the global averaged value, we can use Horovod allreduce.

An example implementing this idea is shown below:
```python
from typing import Optional

try:
    import horovod.torch as hvd
    WITH_HVD = True
except ImportError:
    WITH_HVD = False
    
Tensor = torch.Tensor  # or tf.Tensor for TensorFlow

def global_avg(x: Tensor, name: Optional[str] = None) -> Tensor:
    return hvd.allreduce(x, name=name) if WITH_HVD else x.item()
    
```

---
## Checkpointing from `rank == 0`
Again, it is important to let only one process do the checkpointing I/O to prevent files from being corrupted from race conditions.

```python
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)