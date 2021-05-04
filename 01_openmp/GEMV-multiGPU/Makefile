include make.inc

name=gemv-omp-target-many-matrices-no-hierachy

all_targets=01-gemv-omp-target-one-matrix.x \
            02-gemv-omp-target-many-matrices.x \
            03-gemv-omp-target-many-matrices-multiGPU.x

all: ${all_targets}

%.x: %.cpp
	${CXX} -std=c++11 ${CXX_FLAGS} ${CXX_OFFLOAD_FLAGS} -o $@ $<

.PHONY : clean
clean :
	rm -f *.x
