#!/bin/bash -l
#COBALT -n 2
#COBALT -t 10
#COBALT -q full-node
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true
CONTAINER=$1

export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8 # GPUs per NODE
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS

MPI_BASE=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/
export LD_LIBRARY_PATH=$MPI_BASE/lib:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo mpirun=$(which mpirun)

echo test mpi
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER python -c "import horovod.torch as hvd;hvd.init();print(hvd.local_rank(),hvd.rank(),hvd.size())"
