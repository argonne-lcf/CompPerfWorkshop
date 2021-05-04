# Scaling Deep Learning Applications

---

#### Contents

- [DDP](./DDP/README.md)
  - [torch_ddp_mnist.py](./DDP/torch_ddp_mnist.py)
- [horovod](./horovod/README.md)
  - [tensorflow](./horovod/tensorflow/README.md)
    - [tf2_hvd_mnist.py](./horovod/tensorflow/tf2_hvd_mnist.py)
  - [torch](./horovod/torch/README.md)
    - [torch_hvd_mnist.py](./horovod/torch/torch_hvd_mnist.py)
- [utils](./utils/README.md)
  - [io.py](./utils/io.py)  (Helper functions for creating datasets, logging metrics, etc.)
  - [parse_args.py](./utils/parse_args.py) (Helper functions for parsing command line arguments)
  - [data_torch.py](./utils/data_torch.py) (Helper functions for dealing with datasets in `torch`)
- [theta](./theta/README.md)
  - [submissions](./theta/submissions)
- [thetaGPU](./thetaGPU/README.md)
  - [submissions](./thetaGPU/submissions)

---

Computational Performance Workshop @ ALCF 2021

Author: Sam Foreman ([foremans@anl.gov](mailto:foremans@anl.gov)), Huihuo Zheng ([huhuo.zheng@anl.gov](mailto:huihuo.zheng@anl.gov)), Corey Adams ([corey.adams@anl.gov](mailto:corey.adams@anl.gov))

This section of the workshop will introduce you to the methods we use to run distributed deep learning training on ALCF resource like Theta and ThetaGPU.

---

### Simple Scaling Comparison

![./images/pytorch_scaling.png](./images/scaling_transparent.png)