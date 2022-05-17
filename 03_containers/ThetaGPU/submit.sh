#!/bin/sh
set -e
cd source

printf "running mpirun -n 4 ./mpi_hello_world... "
mpirun -n 4 ./mpi_hello_world
echo done

printf "running python3 mpi_hello_world.py... "
mpirun -n 4 python3 mpi_hello_world.py
echo done
