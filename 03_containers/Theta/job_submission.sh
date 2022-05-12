#!/bin/bash
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -n 2
#COBALT -A datascience
#COBALT --attrs filesystem=theta-fs0,home

RANKS_PER_NODE=4

# pass container as first argument to script
CONTAINER=$1

# Use Cray's Application Binary Independent MPI build
module swap cray-mpich cray-mpich-abi


# Only needed when interactive debugging
#module swap PrgEnv-intel PrgEnv-cray; module swap PrgEnv-cray PrgEnv-intel

export ADDITIONAL_PATHS="/opt/cray/diag/lib:/opt/cray/ugni/default/lib64/:/opt/cray/udreg/default/lib64/:/opt/cray/xpmem/default/lib64/:/opt/cray/alps/default/lib64/:/opt/cray/wlm_detect/default/lib64/"

# in order to pass environment variables to a Singularity container create the
# variable with the SINGULARITYENV_ prefix
export SINGULARITYENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATHS"

TOTAL_RANKS=$(( $COBALT_JOBSIZE * $RANKS_PER_NODE ))

# show that the app is running agains the host machines Cray libmpi.so not the
# one inside the container
BINDINGS="-B /opt -B /etc/alternatives"

# run my containner like an application
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE singularity exec $BINDINGS $CONTAINER /usr/source/mpi_hello_world
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE singularity exec $BINDINGS $CONTAINER python3 /usr/source/mpi_hello_world.py
