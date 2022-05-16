# Containers on Theta (KNL)

## Example `DockerFile`

We include example build source code here: [Local Example Source](/03_containers_AT/03_containers/Local/). This includes an example [DockerFile](/03_containers_AT/03_containers/Local/Dockerfile) which we will describe line-by-line below.

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

Next we copy the [source/](/03_containers_AT/03_containers/Local/source) code examples from our repo (paths are with respect to the location of the `DockerFile`) into our containers filesystem and build the C-file into a binary we can later execute on Theta.

```DockerFile
ENTRYPOINT ["/usr/submit.sh"]
```

In Docker (and Singularity) you can simply "run" a container if an entry point is defined, so calling `docker run <container-name>` in this recipe executes our `submit.sh` script. Otherwise we can be more explicit can call any binary in our container using `docker exec <container-name> <command>`.




