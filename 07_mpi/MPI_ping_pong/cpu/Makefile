#MPICOMP = mpicc
MPICOMP = /soft/thetagpu/hpc-sdk/Linux_x86_64/21.3/comm_libs/mpi/bin/mpicc

pp: ping_pong.o
	$(MPICOMP) ping_pong.o -o pp

ping_pong.o: ping_pong.c
	$(MPICOMP) -c ping_pong.c

.PHONY: clean

clean:
	rm -f pp *.o
