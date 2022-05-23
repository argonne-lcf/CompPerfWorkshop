# Containers on Theta(GPU)

On Theta(GPU), container creation can be achieved by following the upstream [README](../README.md) using Docker, or using a Singularity recipe file and building on a Theta(GPU) worker node.

## Docker on ThetaGPU

If you followed the `Dockerfile` instructions, using the Theta(GPU) specific `Dockerfile_thetagpu` you can build your container for theta using:
```bash
singularity build <image_name> docker://<username>/<repo_name>:<tag>
# using tutorial example
singularity build my_image.simg docker://jtchilders/alcf_cwp_example:thetagpu
```

![singularity_build_thetagpu](../README_media/singularity_build_thetagpu.gif)

Then you can submit a job to Theta(GPU) using
```bash
module load cobalt/cobalt-gpu
qsub -A <project-name> job_submission_thetagpu.sh ./my_image.simg
```

The output should look like this:
```
C++ MPI
Hello world from processor thetagpu12, rank 4 out of 16 processors
Hello world from processor thetagpu12, rank 7 out of 16 processors
Hello world from processor thetagpu12, rank 1 out of 16 processors
Hello world from processor thetagpu12, rank 5 out of 16 processors
Hello world from processor thetagpu12, rank 6 out of 16 processors
Hello world from processor thetagpu12, rank 0 out of 16 processors
Hello world from processor thetagpu12, rank 2 out of 16 processors
Hello world from processor thetagpu12, rank 3 out of 16 processors
Hello world from processor thetagpu18, rank 14 out of 16 processors
Hello world from processor thetagpu18, rank 15 out of 16 processors
Hello world from processor thetagpu18, rank 13 out of 16 processors
Hello world from processor thetagpu18, rank 8 out of 16 processors
Hello world from processor thetagpu18, rank 9 out of 16 processors
Hello world from processor thetagpu18, rank 11 out of 16 processors
Hello world from processor thetagpu18, rank 12 out of 16 processors
Hello world from processor thetagpu18, rank 10 out of 16 processors
Python MPI
Hello world from processor thetagpu18, rank 13 out of 16 processors
Hello world from processor thetagpu18, rank 8 out of 16 processors
Hello world from processor thetagpu18, rank 9 out of 16 processors
Hello world from processor thetagpu18, rank 14 out of 16 processors
Hello world from processor thetagpu18, rank 15 out of 16 processors
Hello world from processor thetagpu18, rank 11 out of 16 processors
Hello world from processor thetagpu18, rank 10 out of 16 processors
Hello world from processor thetagpu18, rank 12 out of 16 processors
Hello world from processor thetagpu12, rank 2 out of 16 processors
Hello world from processor thetagpu12, rank 5 out of 16 processors
Hello world from processor thetagpu12, rank 0 out of 16 processors
Hello world from processor thetagpu12, rank 6 out of 16 processors
Hello world from processor thetagpu12, rank 4 out of 16 processors
Hello world from processor thetagpu12, rank 1 out of 16 processors
Hello world from processor thetagpu12, rank 7 out of 16 processors
Hello world from processor thetagpu12, rank 3 out of 16 processors
```

## Building using Singularity Recipes

While building using Docker on your local machine tends to be the easier method. There are sometimes reasons to build in the environment of the supercomputer. In this case, one can build a singularity container on ThetaGPU in an interactive session on a compute (or worker) node. First a recipe file is needed, here is an example [singularity definition](./mpi.def) file. 

Detailed directions for recipe construction are available on the [Singularity Recipe Page](https://sylabs.io/guides/2.6/user-guide/container_recipes.html).

## Example Singularity definition file

Here we have defined the base image from which to `bootstrap` our container. We are using an image from Docker Hub, `ubuntu:20.04`.

```singularity
Bootstrap: docker
From: ubuntu:20.04
```

The `%files` section lists files to copy from the host system (left path) to the container filesystem (right path)prior to build time.

```singularity
%files
	../Local/source/* /usr/source/
	../Local/submit.sh /usr/
```

The `%environment` section defines environment variables that will be available to the container at runtime.

```singularity
%environment
	export PATH=$PATH:/mpich/install/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mpich/install/lib
```

The `%post` section executes within the container at build time on top of our `ubuntu:20.04` operating system. The `%post` section is therefore the place to perform installations of custom apps with syntax similar to BASH. We won't review these commands here as they are similar to what was covered in the upstream [README](../README.md).

```singularity
%post
	#### INSTALL BASE PACKAGES NEEDED FOR MPI APPLICATIONS AND PYTHON3 ####
	DEBIAN_FRONTEND=noninteractive
	apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive \
	&& apt-get install -y build-essential libfabric-dev libibverbs-dev gfortran wget \
	&& apt-get install -y python3 python3-distutils python3-pip gcc

	#### DOWNLOAD AND INSTALL MPICH AND MPI4PY ####
	# Source is available at http://www.mpich.org/static/downloads/
	# See installation guide of target MPICH version
	# Ex: https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2-installguide.pdf
	# These options are passed to the steps below
	OPENMPI_VERSION_A="4.0"
	OPENMPI_VERSION_B="4.0.5"
	OPENMPI_CONFIGURE_OPTIONS="--prefix=/openmpi/install --disable-wrapper-rpath --disable-wrapper-runpath"
	OPENMPI_MAKE_OPTIONS="-j"
	mkdir -p openmpi
	cd /openmpi
	wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_A}/openmpi-${OPENMPI_VERSION_B}.tar.gz
	tar xfz openmpi-${OPENMPI_VERSION_B}.tar.gz  --strip-components=1
   ./configure ${OPENMPI_CONFIGURE_OPTIONS}
   make install ${OPENMPI_MAKE_OPTIONS}
	
	export PATH=$PATH:/openmpi/install/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/openmpi/install/lib

	pip install mpi4py

	#### BUILD FILES ####
	chmod +x /usr/submit.sh
	mpicc -o /usr/source/mpi_hello_world /usr/source/mpi_hello_world.c
```

The `%runscript` section defines actions for the container to take when it is executed using `singularity run <container_name>`.

```singularity
%runscript
	exec /usr/submit.sh "$@"
```

The `%labels` section allows for custom metadata to be added to the container.

```singularity
%labels
        MAINTAINER Aditya atanikanti@anl.gov
```

The `%help` section can be used to define how to build and run the container.

```singularity
%help
    	This is container is used to illustrate a mpi based def file to build a container running python and c programs. To build the container use singularity build --fakeroot mpi.sif mpi.def
```

## Build Singularity container on ThetaGPU compute

After logging on to Theta login nodes, launch an interactive job using the attrs `fakeroot=true`, `pubnet=true` and specifying the filesystems `filesystems=home,theta-fs0`.

```bash
# on Theta login node, must load cobalt-gpu module to submit jobs to ThetaGPU
module load cobalt/cobalt-gpu
qsub -I -n 1 -t 01:00:00 -q single-gpu -A <project_name> --attrs fakeroot=true:pubnet=true:filesystems=home,theta-fs0
```

Before building the container make sure the ThetaGPU compute nodes have access to external resources, this is achieved by setting the `http_proxy` and `https_proxy` variables
```bash
# setup network proxy to reach outside world
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
```

Now build the container using `--fakeroot` where `<def_filename>.def` is the definition file we have defined in the example above and `<image_name>.sif` is the user defined image file name
```bash
# important you run this in the proper path because the file copies in
# the `%files` section of the recipe uses relative paths on the host.
cd /path/to/CompPerWorkshop/03_containers/ThetaGPU
singularity build --fakeroot <image_name>.sif <def_filename>.def 
```

## Run Singularity container on ThetaGPU compute

An example job submission script is here: [job_submission_thetagpu.sh](./job_submission_thetagpu.sh).

First we define our job and our script takes the container name as an input parameter.

```bash
#!/bin/bash -l
#COBALT -n 1
#COBALT -t 00:10:00
#COBALT -q single-gpu
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true
CONTAINER=$1
```

Enable network access at run time by setting the proxy.

```bash
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
```

Setup our MPI settings, figure out number of nodes `NODES` and fix number of process per node `PPN` and multiply to get total MPI ranks `PROCS`.

```bash
NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=8 # GPUs per NODE
PROCS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  PROCS=$PROCS
```

The OpenMPI installed on ThetaGPU must be used for MPI to properly run across nodes. Here the library path is added to `SINGULARITYENV_LD_LIBRARY_PATH`, which will be used by Singularity to set the container's `LD_LIBRARY_PATH` and therefore tell our executables where to find the MPI libraries.

```bash
MPI_BASE=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/
export LD_LIBRARY_PATH=$MPI_BASE/lib:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo mpirun=$(which mpirun)
```

Finally the exectuable is launched.

```bash
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER /usr/source/mpi_hello_world
```

The job can be submitted using:
```bash
qsub -A <project-name> job_submission_thetagpu.sh /path/to/my_image.sif

The output should look like this:
```bash
C++ MPI
Hello world from processor thetagpu02, rank 12 out of 16 processors
Hello world from processor thetagpu02, rank 8 out of 16 processors
Hello world from processor thetagpu02, rank 10 out of 16 processors
Hello world from processor thetagpu02, rank 11 out of 16 processors
Hello world from processor thetagpu02, rank 13 out of 16 processors
Hello world from processor thetagpu02, rank 9 out of 16 processors
Hello world from processor thetagpu02, rank 14 out of 16 processors
Hello world from processor thetagpu02, rank 15 out of 16 processors
Hello world from processor thetagpu01, rank 0 out of 16 processors
Hello world from processor thetagpu01, rank 1 out of 16 processors
Hello world from processor thetagpu01, rank 2 out of 16 processors
Hello world from processor thetagpu01, rank 3 out of 16 processors
Hello world from processor thetagpu01, rank 4 out of 16 processors
Hello world from processor thetagpu01, rank 5 out of 16 processors
Hello world from processor thetagpu01, rank 6 out of 16 processors
Hello world from processor thetagpu01, rank 7 out of 16 processors
Python MPI
Hello world from processor thetagpu02, rank 9 out of 16 processors
Hello world from processor thetagpu02, rank 10 out of 16 processors
Hello world from processor thetagpu02, rank 11 out of 16 processors
Hello world from processor thetagpu02, rank 15 out of 16 processors
Hello world from processor thetagpu02, rank 13 out of 16 processors
Hello world from processor thetagpu02, rank 8 out of 16 processors
Hello world from processor thetagpu02, rank 12 out of 16 processors
Hello world from processor thetagpu02, rank 14 out of 16 processors
Hello world from processor thetagpu01, rank 7 out of 16 processors
Hello world from processor thetagpu01, rank 3 out of 16 processors
Hello world from processor thetagpu01, rank 1 out of 16 processors
Hello world from processor thetagpu01, rank 4 out of 16 processors
Hello world from processor thetagpu01, rank 5 out of 16 processors
Hello world from processor thetagpu01, rank 6 out of 16 processors
Hello world from processor thetagpu01, rank 0 out of 16 processors
Hello world from processor thetagpu01, rank 2 out of 16 processors
```

## Pre-existing Images for Deep Learning

There are several containers on ThetaGPU that will help you get started with deep learning experiments that can efficiently use the A100 GPUs. We have different optimized container for DL here `ls /lus/theta-fs0/software/thetagpu/nvidia-containers/`

To build on top of a pre-existing container you can simply build on top of the preexisting container by changing the the definition file as follows and installing your modules and dependencies. See [bootstap.def](./bootstrap.def) and an explanation of that file below

```singularity
Bootstrap: localimage
From: /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_21.08-py3.simg
```
The base image is a local singularity image

```singularity
%post
	#### INSTALL YOUR PACKAGES NEEDED FOR YOUR APPLICATION ####
	pip install sklearn
```
In the `%post` section install any package you wish to use in the container

```bash
singularity build --fakeroot bootstrap.def bootstrap.sif
mpirun -np 1 singularity run bootstrap.sif 
```
Here we build and run our container directly on the ThetaGPU compute node and we should see an output like below

```bash
2022-05-17 12:11:15.755958: I tensorflow/stream_executor/platform/default/dso_loader.cc:54] Successfully opened dynamic library libcudart.so.11.0
2022-05-17 12:11:19.144265: W tensorflow/stream_executor/platform/default/dso_loader.cc:65] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs
2022-05-17 12:11:19.144303: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-05-17 12:11:19.144333: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: thetagpu16
2022-05-17 12:11:19.144340: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: thetagpu16
2022-05-17 12:11:19.144381: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:200] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
2022-05-17 12:11:19.144422: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:204] kernel reported version is: 470.82.1
2022-05-17 12:11:19.147914: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2022-05-17 12:11:19.333080: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2022-05-17 12:11:19.352126: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2245815000 Hz
Accuracy at step 0: 0.216
Accuracy at step 1: 0.098
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
Accuracy at step 4: 0.098
Accuracy at step 5: 0.098
Accuracy at step 6: 0.098
Accuracy at step 7: 0.098
Accuracy at step 8: 0.098
Accuracy at step 9: 0.098
```

You can also submit a job from the service node as we have done before using the [job_submission_thetagpudl.sh](./job_submission_thetagpudl.sh) script
