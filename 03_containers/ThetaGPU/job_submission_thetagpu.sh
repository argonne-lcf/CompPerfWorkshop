#!/bin/bash -l
#COBALT -n 1
#COBALT -t 00:10:00
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true
CONTAINER=$1

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

mpirun -np 1 singularity exec $CONTAINER /usr/source/mpi_hello_world
mpirun -np 1 singularity exec $CONTAINER python3 /usr/source/mpi_hello_world.py

