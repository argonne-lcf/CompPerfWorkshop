#!/bin/bash -l
#COBALT -n 1
#COBALT -t 00:10:00
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
#CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_21.08-py3.simg
#singularity exec --nv $CONTAINER python /usr/local/lib/python3.8/dist-packages/tensorflow/python/debug/examples/debug_mnist.py
mpirun -np 1 singularity run bootstrap.sif