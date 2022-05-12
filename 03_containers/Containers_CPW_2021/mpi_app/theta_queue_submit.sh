#!/bin/bash
#COBALT -t 30
#COBALT -q debug-cache-quad
#COBALT -n 2

RANKS_PER_NODE=4

# pass container as first argument to script
CONTAINER=$1

# Use Cray's Application Binary Independent MPI build
module swap cray-mpich cray-mpich-abi

# Output current modules being used (for debugging)
module list

# Only needed when interactive debugging
#module swap PrgEnv-intel PrgEnv-cray; module swap PrgEnv-cray PrgEnv-intel

# Need /opt/cray/diag/lib for the following libraries: libifport.so.5, libifcore.so.5, libimf.so, libsvml.so, libintlc.so.5
# Need /opt/cray/ugni/default/lib64/ for the following libraries: libugni.so.0
# Need /opt/cray/udreg/default/lib64/ for the following libraries: libudreg.so.0
# Need /opt/cray/xpmem/default/lib64/ for the following libraries: libxpmem.so.0
# Need /opt/cray/alps/default/lib64/ for the following libraries: libalpsutil.so.0
# Need /opt/cray/wlm_detect/default/lib64/ for the following libraries: libwlm_detect.so.0
export ADDITIONAL_PATHS="/opt/cray/diag/lib:/opt/cray/ugni/default/lib64/:/opt/cray/udreg/default/lib64/:/opt/cray/xpmem/default/lib64/:/opt/cray/alps/default/lib64/:/opt/cray/wlm_detect/default/lib64/"

# in order to pass environment variables to a Singularity container create the
# variable with the SINGULARITYENV_ prefix
export SINGULARITYENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATHS"

# print to log file for debug
echo $SINGULARITYENV_LD_LIBRARY_PATH

TOTAL_RANKS=$(( $COBALT_JOBSIZE * $RANKS_PER_NODE ))

# this simply runs the command 'ldd /myapp/pi' inside the container and should
# show that the app is running agains the host machines Cray libmpi.so not the
# one inside the container
BINDINGS="-B /opt -B /etc/alternatives"
aprun -n 1 -N 1 singularity exec $BINDINGS $CONTAINER bash -c "echo \$LD_LIBRARY_PATH"
aprun -n 1 -N 1 singularity exec $BINDINGS $CONTAINER bash -c "ldd /myapp/pi"

# run my containner like an application, which will run '/myapp/pi'
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE singularity run $BINDINGS $CONTAINER