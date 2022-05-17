# Containers on Theta(GPU)

## Singularity on ThetaGPU

To build a singularity container on ThetaGPU, you will need to launch an interactive job and build the container on ThetaGPU compute nodes. But first you'll need a [singularity definition](./mpi.def) file. See below for explanation of the file.

## Example Singularity definition file

```singularity
Bootstrap: docker
From: ubuntu:20.04
```
Here we have defined the base image to build our singularity container. We are using the Dockerhub's ubuntu:20.04 base image

```singularity
%files
	../local/source/* /usr/source/
	../local/submit.sh /usr/
```
The `%files` section copies some files into the container from the host system at build time

```singularity
%environment
	export PATH=$PATH:/mpich/install/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mpich/install/lib
```
The `%environment` section defines some environment variables that will be available to the container at runtime

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
	MPICH_VERSION="3.3"
	MPICH_CONFIGURE_OPTIONS="--prefix=/mpich/install --disable-wrapper-rpath"
	MPICH_MAKE_OPTIONS="-j 4"
	mkdir -p mpich \
	&& cd /mpich \
	&& wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
	&& tar xfz mpich-${MPICH_VERSION}.tar.gz  --strip-components=1 \
	&& ./configure ${MPICH_CONFIGURE_OPTIONS} \
	&& make install ${MPICH_MAKE_OPTIONS}
	export PATH=$PATH:/mpich/install/bin
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mpich/install/lib

	pip install mpi4py

	#### BUILD FILES ####
	chmod +x /usr/submit.sh
	mpicc -o /usr/source/mpi_hello_world /usr/source/mpi_hello_world.c
```
The `%post` section executes within the container at build time after the base OS has been installed. The %post section is therefore the place to perform installations of custom apps.


```singularity
%runscript
	exec /usr/submit.sh "$@"
```
The `%runscript` section defines actions for the container to take when it is executed

```singularity
%labels
        MAINTAINER Aditya atanikanti@anl.gov
```
The `%labels` section allows for custom metadata to be added to the container

```singularity
%help
    	This is container is used to illustrate a mpi based def file to build a container running python and c programs. To build the container use singularity build --fakeroot mpi.sif mpi.def
```
The `%help` section can be used to define how to build and run the container.

## Build Singularity container on ThetaGPU compute

After logging on to Theta login nodes, ssh to thetagpusn1 and launch an interactive job using the attrs `--fakeroot`, `--pubnet` and specifying the filesystems `--filesystems` as shown
```bash
ssh thetagpusn1
qsub -I -n 1 -t 01:00:00 -q single-gpu -A <project_name> --attrs fakeroot=true:pubnet=true:filesystems=home,theta-fs0
```

Before we build our container we need to make sure the thetagpu computes have access to external resources, this is achieved by defining the http_proxy and https_proxy variables
```bash
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
```

Now we can build our container using `--fakeroot` where <def_filename>.def is the definition file we have defined in our example above and <image_name>.sif is the user defined image file name
```bash
username@thetagpu16: singularity build --fakeroot <image_name>.sif <def_filename>.def 
```

## Run Singularity container on ThetaGPU compute

We can now either run the container on the compute node as shown below
```bash
mpirun -np 1 singularity exec <image_name>.sif /usr/source/mpi_hello_world
mpirun -np 1 singularity exec <image_name>.sif python3 /usr/source/mpi_hello_world.py
```

or from the service node using the script [job_submission_thetagpu.sh](./job_submission_thetagpu.sh) which is explained below
```bash
username@thetagpusn1:qsub /path/to/CompPerfWorkshop/03_containers/ThetaGPU/job_submission_thetagpu.sh </path/to/image_name>.sif
```

## Example `job_submission_thetagpu.sh`

```bash
#!/bin/bash -l
#COBALT -n 1
#COBALT -t 00:10:00
#COBALT -q single-gpu
#COBALT -A <project_name>
#COBALT --attrs filesystems=home,theta-fs0:pubnet=true
CONTAINER=$1
```
Here we have defined the job submission parameters that are needed to submit a job on ThetaGPU, we also pass the container as an argument to the submission script

```bash
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
```
This allows for internet access on the computes and is a prerequisite to run the container

```bash
mpirun -np 1 singularity exec $CONTAINER /usr/source/mpi_hello_world
mpirun -np 1 singularity exec $CONTAINER python3 /usr/source/mpi_hello_world.py
```
Here we run the the singularity container with our example mpi codes using mpirun on ThetaGPU compute nodes

The output should look like this:
```bash
Hello world from processor thetagpu16, rank 0 out of 1 processors
Hello world from processor thetagpu16, rank 0 out of 1 processors
```

## Pre-existing Images for Deep Learning

There are several containers on ThetaGPU that will help you get started with deep learning experiments that can efficiently use the A100 GPUs. We have different optimized container for DL here `ls /lus/theta-fs0/software/thetagpu/nvidia-containers/`

To build on top of a pre-existing container you can simply build on top of the preexisting container by changing the the definition file as follows and installing your modules and dependencies. See [bootstap.def](./bootstrap.def)

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