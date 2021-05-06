#CUCOMP  = nvcc  ## to be used when OMPI_DIR or MPICH_DIR are used below
CUCOMP  = /soft/thetagpu/hpc-sdk/Linux_x86_64/21.3/cuda/bin/nvcc
CUFLAGS = 
OMPI_DIR = /lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.0_nvhpc-21.3
MPICH_DIR= /projects/Performance/chunduri/ThetaGPU/mpich_build_ofi_cuda/
NVMPI_DIR= /soft/thetagpu/hpc-sdk/Linux_x86_64/21.3/comm_libs/mpi
INCLUDES  = -I$(OMPI_DIR)/include
LIBRARIES = -L$(OMPI_DIR)/lib  -lmpi
#INCLUDES  = -I$(MPICH_DIR)/include
#LIBRARIES = -L$(MPICH_DIR)/lib  -lmpi
#INCLUDES   = -I$(NVMPI_DIR)/include
#LIBRARIES  = -L$(NVMPI_DIR)/lib -lmpi

pp_cuda_aware: ping_pong_cuda_aware.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) ping_pong_cuda_aware.o -o pp_cuda_aware

ping_pong_cuda_aware.o: ping_pong_cuda_aware.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c ping_pong_cuda_aware.cu

.PHONY: clean

clean:
	rm -f pp_cuda_aware *.o
