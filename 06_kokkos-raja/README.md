# Kokkos
```
cd
mkdir kokkos
cd kokkos
```
##
## Installing Kokkos on login node
##
```
git clone https://github.com/kokkos/kokkos.git

mkdir build

cd build

module load intel
module load gcc
module load cmake

export CRAYPE_LINK_TYPE=dynamic

cmake ../kokkos/ \
      -DCMAKE_BUILD_TYPE=Release\
      -DCMAKE_CXX_COMPILER=CC\
      -DKokkos_ARCH_KNL=On\
      -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
      -DKokkos_ENABLE_OPENMP=On\
      -DKokkos_ENABLE_SERIAL=On\
      -DKokkos_ENABLE_TESTS=Off
      
make -j
make install

export CMAKE_PREFIX_PATH=$(pwd)/install:$CMAKE_PREFIX_PATH
```
##
## Installing Kokkos Exercises on login node
##
```
git clone https://github.com/kokkos/kokkos-tutorials.git

cd ./kokkos-tutorials/Exercises/04/Solution

mkdir build
cd build
cmake ../

make

// on interactive node run
// aprun --cc depth -d 256 -j 4 ./04_Exercise -nrepeat 200  -N 14 -M 12

// change Kokkos::LayoutLeft to Kokkos::LayoutRight to improve bandwidth
make

// on interactive node run
// aprun --cc depth -d 256 -j 4 ./04_Exercise -nrepeat 200  -N 14 -M 12
```
##
## Log into interactive KNL node
##
```
qsub -I -A comp_perf_workshop -n 1 -t 60 -q debug-cache-quad
module load intel
module load gcc
```
##
## aprun affinity tool
##
```
https://github.com/argonne-lcf/GettingStarted/blob/master/Examples/theta/affinity/main.cpp

CC -qopenmp main.cpp

# KNL has 64 cores with 4 threads per core, i.e. in total 256 threads
aprun lscpu

# run with 4 threads per core
aprun --cc depth -d 256 -j 4 ./a.out 
# run with 1 thread per core
aprun --cc depth -d 64 -j 1 ./a.out
```
