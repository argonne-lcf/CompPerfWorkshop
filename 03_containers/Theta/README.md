# Containers on Theta (KNL)

## Singularity on Theta

To build a singularity container we can either
* Build a Singularity container from a Docker container on the login nodes. To build a Docker container from scratch you need to install Docker on your local machine, write a Dockerfile, build the Docker imag/container, and finally publish it to DockerHub. We illustrate this below with examples. Or
* Build a Singularity container directly on ThetaGPU computes. We achieve this by writing a Singularity Definition file and build the singularity container on ThetaGPU compute nodes. See [here](../ThetaGPU) for examples.


## Build Singularity image from DockerHub

Now that we have a docker image on DockerHub, we can build our singularity container using the docker image as a source using `sinularity build <image_name> docker://..`

```bash
singularity build <image_name> docker://<username>/<repository_name>:<tag>
```

Here `image_name` is user defined & usually ends with `.sif` or `.simg`.

## Run Singularity container on Theta

```bash
qsub /path/to/CompPerfWorkshop/03_containers/Theta/job_submission_theta.sh </path/to/image_name>
```

## Example `job_submission_theta.sh`

```bash
#!/bin/bash
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -n 2
#COBALT -A <project_name>
#COBALT --attrs filesystems=theta-fs0,home
RANKS_PER_NODE=4
CONTAINER=$1
```

Here we have defined the job submission parameters that are needed to submit a job on Theta, the number of ranks per node and we also pass the container as an argument to the submission script

```bash
# Use Cray's Application Binary Independent MPI build
module swap cray-mpich cray-mpich-abi
# Only needed when interactive debugging
#module swap PrgEnv-intel PrgEnv-cray; module swap PrgEnv-cray PrgEnv-intel

export ADDITIONAL_PATHS="/opt/cray/diag/lib:/opt/cray/ugni/default/lib64/:/opt/cray/udreg/default/lib64/:/opt/cray/xpmem/default/lib64/:/opt/cray/alps/default/lib64/:/opt/cray/wlm_detect/default/lib64/"

# in order to pass environment variables to a Singularity container create the
# variable with the SINGULARITYENV_ prefix
export SINGULARITYENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATHS"

# show that the app is running agains the host machines Cray libmpi.so not the
# one inside the container
BINDINGS="-B /opt -B /etc/alternatives"
```

Here we define all the prerequisite settings needed to run singularity on Theta compute nodes

```bash
TOTAL_RANKS=$(( $COBALT_JOBSIZE * $RANKS_PER_NODE ))
# run my containner like an application
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE singularity exec $BINDINGS $CONTAINER /usr/source/mpi_hello_world
aprun -n $TOTAL_RANKS -N $RANKS_PER_NODE singularity exec $BINDINGS $CONTAINER python3 /usr/source/mpi_hello_world.py
```

Here we run the the singularity container with our example mpi codes using aprun on Theta compute nodes

The output should look like this:
```
Hello world from processor nid00020, rank 2 out of 8 processors
Hello world from processor nid00020, rank 3 out of 8 processors
Hello world from processor nid00020, rank 0 out of 8 processors
Hello world from processor nid00020, rank 1 out of 8 processors
Hello world from processor nid00021, rank 6 out of 8 processors
Hello world from processor nid00021, rank 7 out of 8 processors
Hello world from processor nid00021, rank 4 out of 8 processors
Hello world from processor nid00021, rank 5 out of 8 processors
Application 26449404 resources: utime ~14s, stime ~8s, Rss ~39912, inblocks ~64022, outblocks ~0
Hello world from processor nid00021, rank 7 out of 8 processors
Hello world from processor nid00021, rank 6 out of 8 processors
Hello world from processor nid00021, rank 5 out of 8 processors
Hello world from processor nid00021, rank 4 out of 8 processors
Hello world from processor nid00020, rank 2 out of 8 processors
Hello world from processor nid00020, rank 3 out of 8 processors
Hello world from processor nid00020, rank 1 out of 8 processors
Hello world from processor nid00020, rank 0 out of 8 processors
Application 26449405 resources: utime ~14s, stime ~8s, Rss ~39392, inblocks ~83290, outblocks ~0
```
