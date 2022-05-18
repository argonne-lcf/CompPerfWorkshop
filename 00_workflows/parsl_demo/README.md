# Parsl Tutorial

This tutorial shows how Parsl can be used to run computational workflows on Theta. Workflows
are written in Python and in this case we launch and manage them from a Jupyter notebook. 

## About Parsl
[Parsl](parsl.readthedocs.io). is a parallel programming library for Python. Parsl augments Python with simple, scalable, and flexible constructs for encoding parallelism. Developers annotate Python functions to specify opportunities for concurrent execution. These annotated functions, called apps, may represent pure Python functions or calls to external applications, whether sequential, multicore (e.g., CPU, GPU, accelerator), or multi-node MPI. Parsl further allows these calls to these apps, called tasks, to be connected by shared input/output data (e.g., Python objects or files) via which Parsl can construct a dynamic dependency graph of tasks.

Parsl is built around a flexible and scalable runtime that allows it to efficiently execute Python programs in parallel. Parsl scripts are portable and can be easily moved between different execution resources: from laptops to supercomputers to clouds. When executing a Parsl program, developers first define a simple Python-based configuration that outlines where and how to execute tasks. Parsl supports various target resources including clouds (e.g., Amazon Web Services and Google Cloud), clusters (e.g., using Slurm, Torque/PBS, HTCondor, Cobalt), and container orchestration systems (e.g., Kubernetes). Parsl scripts can scale from a single core on a single computer through to hundreds of thousands of cores across many thousands of nodes on a supercomputer.

More information is available in the [Parsl documentation](https://parsl.readthedocs.io/en/stable/).

# Tutorial Setup
Parsl is a Python library which can be deployed in user space via pip (i.e., `pip install parsl`). 

Before starting the tutorial we suggest creating a new conda environment that can be used from the ALCF Jupyter system. 

```
conda create -n parsl-tutorial
source activate parsl-tutorial
conda install jupyter nb_conda ipykernel
pip install parsl
```

To make the environment accessible to Jupyter type the following

```
python -m ipykernel install --user --name parsl-tutorial
```

You can then open Jupyter in your browser and remember to change the Python kernel to parsl-tutorial

https://jupyter.alcf.anl.gov/

You can clone this repository and follow along with the notebook.
