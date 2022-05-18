## Organization

- [`DDP/`](./DDP/README.md) Contains simple implementation using PyTorch's native [Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) library
    - [`conf/config.yaml`](./DDP/conf/config.yaml) [Hydra](https://hydra.cc) config file specifying configuration options for running an experiment.
    - [`main.py`](./DDP/main.py): Entry point for training and evaluating model using DDP
- [`deepspeed/`](./deepspeed/README.md) Implementation using Microsoft's [DeepSpeed](https://www.microsoft.com/en-us/research/project/deepspeed) library
    - [`conf/config.yaml`](./deepspeed/conf/config.yaml)
    - [`main.py`](./deepspeed/main.py): DeepSpeed entry 
- [`horovod/`](./horovod/README.md)
    - [`tensorflow/`](./horovod/tensorflow/README.md)
        - [`conf/config.yaml`](./horovod/tensorflow/conf/config.yaml)
        - [`main.py`](./horovod/tensorflow/main.py)
    - [`torch/`](./horovod/torch/README.md)
        - [`conf/config.yaml`](./horovod/torch/conf/config.yaml)
        - [`main.py`](./horovod/torch/main.py)
        