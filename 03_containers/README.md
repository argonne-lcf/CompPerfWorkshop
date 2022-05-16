# Containers at ALCF
Contributors: Taylor Childers, Aditya Tanikanti, < add past contributors >

At ALCF we support [Singularity](https://sylabs.io/guides/3.5/user-guide/index.html) containers which is a container technology built for supercomputers with securtiy in mind. Typically we recommend users build containers via [Docker](https://docs.docker.com/) containers. Below you'll find links to instructions for each of our systems.

We will not repeat the detailed instructions for building docker containers, but do provide system specific examples of what a `DockerFile` should look like. 
* General Docker documentation can be found here: https://docs.docker.com/
* Specifics on building docker container recipes using `DockerFile` can be found here: https://docs.docker.com/engine/reference/builder/

The trickiest parts of building containers for ALCF systems is ensuring proper MPI support and GPU driver compatibility.

# System Specific Instructions

* [Containers on Theta (KNL)](Theta/)
* [Containers on Theta (GPU)](ThetaGPU/)
