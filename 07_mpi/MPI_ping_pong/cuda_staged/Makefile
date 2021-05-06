#CUCOMP  = nvcc
CUCOMP  = /soft/thetagpu/hpc-sdk/Linux_x86_64/21.3/cuda/bin/nvcc
CUFLAGS = 
#OMPI_DIR = /lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.0_nvhpc-21.3
NVMPI_DIR= /soft/thetagpu/hpc-sdk/Linux_x86_64/21.3/comm_libs/mpi
#INCLUDES  = -I$(OMPI_DIR)/include
#LIBRARIES = -L$(OMPI_DIR)/lib -lmpi
INCLUDES   = -I$(NVMPI_DIR)/include
LIBRARIES  = -L$(NVMPI_DIR)/lib -lmpi

pp_cuda_staged: ping_pong_cuda_staged.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) ping_pong_cuda_staged.o -o pp_cuda_staged

ping_pong_cuda_staged.o: ping_pong_cuda_staged.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c ping_pong_cuda_staged.cu

.PHONY: clean

clean:
	rm -f pp_cuda_staged *.o
