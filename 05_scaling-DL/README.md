# Scaling Deep Learning Applications

Author: Sam Foreman ([foremans@anl.gov](mailto:foremans@anl.gov))

This section of the workshop will introduce you to the methods we use to run distributed deep learning training on ALCF resource like Theta and ThetaGPU.

We provide examples for:

1. [Horovod](./horovod/README.md)
   1. [TensorFlow](./horovod/tensorflow/README.md)
   2. [PyTorch](./horovod/torch/README.md)

1. [DDP (PyTorch)](./DDP/README.md)

## Contents

<pre>
.
├── DDP
│   ├── README.md
│   ├── pl_cifar10.py
│   └── torch_ddp_cifar10.py 				# script for training with DDP + PyTorch
├── horovod
│   ├── tensorflow							
│   │   ├── README.md
│   │   └── tf2hvd_mnist.py					# script for training with Horovod + TensorFlow
│   ├── torch
│   │   ├── README.md
│   │   └── torch_cifar10_hvd.py			# script for training with Horovod + PyTorch
│   └── README.md
├── theta									# directory containing scripts for submitting jobs to Theta
│   ├── submissions
│   │   ├── qsub_keras_cifar10_scale.sh
│   │   ├── qsub_pytorch_cifar10.sh
│   │   ├── qsub_pytorch_mnist_scale.sh
│   │   ├── qsub_tensorflow_cifar10.sh
│   │   └── qsub_tensorflow_mnist.sh
│   └── theta.md
├── thetaGPU								# directory containing scripts for submitting jobs to ThetaGPU
│   ├── submissions
│   │   ├── qsub_keras_cifar10.sh
│   │   ├── qsub_keras_mnist.sh
│   │   ├── qsub_pytorch_cifar10.sh
│   │   ├── qsub_pytorch_mnist.sh
│   │   ├── qsub_tensorflow_cifar10.sh
│   │   └── qsub_tensorflow_mnist.sh
│   └── thetaGPU.md
├── utils									# directory of helper functions
│   ├── __init__.py		
│   ├── io.py								# contains `Logger` object for printing nicely formatted logs
│   └── parse_args.py						# contains function for passing common command line arguments
</pre>