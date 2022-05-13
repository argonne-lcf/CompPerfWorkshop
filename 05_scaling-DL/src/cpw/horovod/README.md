    
# Horovod <img src="assets/horovod-logo.png" width="50" style="vertical-align:baseline;" align="left"> 
Here we describe how to use Horovod for distributed data-parallel training using both `pytorch` and `tensorflow`.

Additional information can be found on [Horovod documentation](https://horovod.readthedocs.io/en/stable/index.html) page.

## Organization
1. [Horovod with Tensorflow](./tensorflow/README.md)
    1. [`tensorflow/main.py`](./tensorflow/main.py)
2. [Horovod with Tensorflow + Keras](./tensorflow/keras/README.md)
    1. [`tensorflow/keras/main.py`](./tensorflow/keras/main.py)
3. [Horovod with PyTorch](./pytorch/README.md)
    1. [`pytorch/main.py`](./pytorch/main.py)

## Overview
The basic procedure of setting up and using Horovod for distributed training is almost identical in both frameworks and consists of the following main steps:

1. Initialize Horovod
2. Assign GPUs to each rank
3. Scale the initial learning rate by the number of workers
4. Distribute Gradients & Broadcast State
    1.  Distribute gradients by wrapping the `optimizer` object with `hvd.DistributedOptimizer`broadcast model weights
    2.  Ensure consistent initialization across workers by broadcasting model weights and optimizer state to all workers from `rank = 0`
5.  Ensure workers are always receiving unique data
6.  Take global averages when calculating `loss`, `acc`, etc using `hvd.allreduce(...)` 
7. Save checkpoints _only_ from chief rank, i.e. `rank = 0` worker to prevent race conditions

See [Horovod with Tensorflow](./tensorflow/README.md) for additional information and details on the specifics of using Horovod with TensorFlow.

