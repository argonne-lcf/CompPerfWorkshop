# Containers on Theta (KNL)

## Singularity on Theta

To build a singularity container we can either
* Build a Singularity container from a Docker container on the login nodes. To build a Docker container from scratch you need to install Docker on your local machine, write a Dockerfile, build the Docker imag/container, and finally publish it to DockerHub. We illustrate this below with examples. Or
* Build a Singularity container directly on ThetaGPU computes. We achieve this by writing a Singularity Definition file and build the singularity container on ThetaGPU compute nodes. See [here](../ThetaGPU) for examples.

## Example `DockerFile`

We include example build source code here: [Local Example Source](../Local/source). This includes an example [DockerFile](../Local/Dockerfile) which we will describe line-by-line below.

```DockerFile
FROM ubuntu:20.04
```
The first line specifies a starting point for our contianer. In this instance, we start from a container that only has Ubuntu version 20.04 installed. However, we could start with any other container available on Docker Hub. We'll build everything else on top of this operating system.

```DockerFile
RUN apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive \
	&& apt-get install -y build-essential libfabric-dev libibverbs-dev gfortran wget \
        && apt-get install -y python3 python3-distutils python3-pip gcc
```

Here we install system packages we need to build and run our code examples using the standard [Ubuntu package manager](https://ubuntu.com/server/docs/package-management#:~:text=The%20apt%20command%20is%20a,upgrading%20the%20entire%20Ubuntu%20system.) `apt`.

```DockerFile
WORKDIR /mpich
# Source is available at http://www.mpich.org/static/downloads/
# See installation guide of target MPICH version
# Ex: https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2-installguide.pdf
# These options are passed to the steps below
ARG MPICH_VERSION="3.3"
ARG MPICH_CONFIGURE_OPTIONS="--prefix=/mpich/install --disable-wrapper-rpath"
ARG MPICH_MAKE_OPTIONS="-j 4"
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
      && tar xfz mpich-${MPICH_VERSION}.tar.gz  --strip-components=1 \
      && ./configure ${MPICH_CONFIGURE_OPTIONS} \
      && make install ${MPICH_MAKE_OPTIONS}
ENV PATH $PATH:/mpich/install/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/mpich/install/lib
```

Here we change our working directory to `/mpich` and then download and install MPI from scratch with some specific build options. You can find the installation documentation [HERE](https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2-installguide.pdf). The key compilation option is the `--disable-wrapper-rpath` which makes it possible to build applications inside the container using this MPI library, but then replace those libraries with the Theta-specific libraries during runtime simply using the `LD_LIBRARY_PATH` environment variable. This is important since Theta uses high-speed network interfaces that require custom drivers and interface libraries to use.

```DockerFile
RUN pip install mpi4py
```

Here we simply install `mpi4py` into our python environment and it will utilize the MPICH we installed.

```DockerFile
WORKDIR /usr
COPY source/* /usr/source/
COPY submit.sh /usr/
RUN chmod +x /usr/submit.sh
RUN mpicc -o /usr/source/mpi_hello_world /usr/source/mpi_hello_world.c
```

Next we copy the [source/](/03_containers/Local/source) code examples from our repo (paths are with respect to the location of the `DockerFile`) into our containers filesystem and build the C-file into a binary we can later execute on Theta.

```DockerFile
ENTRYPOINT ["/usr/submit.sh"]
```

In Docker (and Singularity) you can simply "run" a container if an entry point is defined, so calling `docker run <container-name>` in this recipe executes our `submit.sh` script. Otherwise we can be more explicit can call any binary in our container using `docker exec <container-name> <command>`.

## Publish Docker Image to DockerHub

To build and publish your docker image to [DockerHub](https://hub.docker.com/) use docker build followed by docker push.

```bash
# in some cases need to login to docker (for Mac OS need to run Docker desktop)
docker login
# build image from DockerFile, include the path to the folder that contains the DockerFile
docker build -t <username>/<repository_name>:<tag> </path/to/CompPerfWorkshop/03_containers/Local/>
# push this image into your docker hub account so it is accessible remotely
docker push <username>/<respository_name>:<tag>
```
username & repository_name are created on [DockerHub](https://hub.docker.com/). 

e.g.

```bash
docker login
docker build -t atanikan/alcftutorial:latest
docker push atanikan/alcftutorial:latest
```

## Build Singularity image from DockerHub

Now that we have a docker image on DockerHub, we can build our singularity container using the docker image as a source using `sinularity build <image_name> docker://..` 

e.g.
```console
thetalogin6: singularity build <image_name>.sif docker://<username>/<repository_name>:<tag>
```

Here image_name is user defined & usually ends with .sif or .img

## Run Singularity container on Theta

```console
thetalogin6: qsub job_submission_theta.sh <image_name>
```

## Example `job_submission_theta.sh`

```bash
#!/bin/bash
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -n 2
#COBALT -A <project_name>
#COBALT --attrs filesystem=theta-fs0,home
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
