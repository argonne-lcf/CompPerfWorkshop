#!/bin/bash -l
#COBALT -n 2
#COBALT -t 10
#COBALT -q full-node
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true
CONTAINER=$1

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

wget https://github.com/horovod/horovod/raw/master/examples/pytorch/pytorch_synthetic_benchmark.py

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8 # GPUs per NODE
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

echo test mpi
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN \
   singularity exec --nv -B $PWD $CONTAINER \
      python $PWD/pytorch_synthetic_benchmark.py
