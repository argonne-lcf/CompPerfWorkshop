# Scaling Deep Learning Applications

---

### Table of Contents

- [DDP](./DDP/README.md)
  - [torch_ddp_cifar10.py](./DDP/torch_ddp_cifar10.py)
- [horovod](./horovod/README.md)
  - [tensorflow](./horovod/tensorflow/README.md)
    - [tf2_hvd_cifar10.py](./horovod/tensorflow/tf2_hvd_cifar10.py)
  - [torch](./horovod/torch/README.md)
    - [torch_hvd_cifar10.py](./horovod/torch/torch_hvd_cifar10.py)
- [utils](./utils/README.md)
  - [io.py](./utils/io.py)  (Helper functions for creating datasets, logging metrics, etc.)
  - [parse_args.py](./utils/parse_args.py) (Helper functions for parsing command line arguments)
- [theta](./theta/README.md)
  - [submissions](./theta/submissions)
- [thetaGPU](./thetaGPU/README.md)
  - [submissions](./thetaGPU/submissions)

---

Computational Performance Workshop @ ALCF 2021

Author: Sam Foreman ([foremans@anl.gov](mailto:foremans@anl.gov)), Huihuo Zheng ([huhuo.zheng@anl.gov](mailto:huihuo.zheng@anl.gov))

This section of the workshop will introduce you to the methods we use to run distributed deep learning training on ALCF resource like Theta and ThetaGPU.

---

### Simple Scaling Comparison

![./images/pytorch_scaling.png](./images/scaling_transparent.png)