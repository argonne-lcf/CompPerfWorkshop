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

echo library path
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER ldd /usr/source/mpi_hello_world

echo C++ MPI
# --nv is required on ThetaGPU to include the NVIDIA drivers from the host OS
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER /usr/source/mpi_hello_world

echo Python MPI
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER python3 /usr/source/mpi_hello_world.py

