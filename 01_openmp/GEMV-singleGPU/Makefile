include make.inc

name=gemv-omp-target-many-matrices-no-hierachy

all_targets=1-gemv-omp.cxx.x \
            2-gemv-omp-target-reduction.cxx.x \
            3-gemv-omp-target-data.cxx.x \
            1-gemv-omp.f.x \
            2-gemv-omp-target-reduction.f.x \
            3-gemv-omp-target-data.f.x

all: ${all_targets}

%.cxx.x: %.cpp
	${CXX} -std=c++11 ${CXX_FLAGS} ${OFFLOAD_FLAGS} -o $@ $<

%.f.x: %.f90
	${FC} ${FC_FLAGS} ${OFFLOAD_FLAGS} -o $@ $<

.PHONY : clean
clean :
	rm -f *.x
