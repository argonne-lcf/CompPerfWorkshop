FROM ubuntu:20.04
MAINTAINER Aditya atanikanti@anl.gov

#### INSTALL BASE PACKAGES NEEDED FOR MPI APPLICATIONS AND PYTHON3 ####
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
	&& DEBIAN_FRONTEND=noninteractive \
	&& apt-get install -y build-essential libfabric-dev libibverbs-dev gfortran wget \
        && apt-get install -y python3 python3-distutils python3-pip gcc

#### DOWNLOAD AND INSTALL MPICH AND MPI4PY ####
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
RUN pip install mpi4py

#### COPY FILES INTO CONTAINER ####
WORKDIR /usr
COPY source/* /usr/source/
COPY submit.sh /usr/
RUN chmod +x /usr/submit.sh
RUN mpicc -o /usr/source/mpi_hello_world /usr/source/mpi_hello_world.c

#### SPECIFY SCRIPT FOR CONTAINER TO RUN ####
ENTRYPOINT ["/usr/submit.sh"]
