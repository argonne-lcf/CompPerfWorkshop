# NVIDIA software tutorial

To get started, on a ThetaGPU allocation, do

```
$ module load nvhpc/21.3
```

Now we can compile and run (with profiling) each of the example codes, which
each perform a DAXPY (double precision a * x + y) in either Fortran or C++:

## Standard Language Parallelism

C++:
```
$ nvc++ -stdpar -o daxpy_stdpar daxpy_stdpar.cpp
$ nsys profile --stats=true ./daxpy_stdpar
```

Fortran:
```
$ nvfortran -stdpar -o daxpy_stdpar daxpy_stdpar.f90
$ nsys profile --stats=true ./daxpy_stdpar
```
## OpenACC

C++:
```
$ nvc++ -acc -o daxpy_acc daxpy_acc.cpp
$ nsys profile --stats=true ./daxpy_acc
```

Fortran:
```
$ nvfortran -acc -o daxpy_acc daxpy_acc.f90
$ nsys profile --stats=true ./daxpy_acc
```

## OpenMP

C++:
```
$ nvc++ -mp=gpu -o daxpy_omp daxpy_omp.cpp
$ nsys profile --stats=true ./daxpy_omp
```

Fortran:
```
$ nvfortran -mp=gpu -o daxpy_omp daxpy_omp.f90
$ nsys profile --stats=true ./daxpy_omp
```

## cuBLAS

C++:
```
$ nvc++ -gpu=managed -cuda -cudalib=cublas -o daxpy_blas daxpy_blas.cpp
$ nsys profile --stats=true ./daxpy_blas
```

Fortran:
```
$ nvfortran -cuda -cudalib=cublas -o daxpy_blas daxpy_blas.f90
$ nsys profile --stats=true ./daxpy_blas
```

## Python (Numba)

We need to get our environment set up first. We'll take advantage of the
TensorFlow conda installation just for ease of use.
```
module load conda/tensorflow
conda activate
conda create -n numba_env
conda activate numba_env
conda install numba
conda install cudatoolkit
```

Now we can run and profile the application:
```
nsys profile --stats=true python daxpy_numba.py
```

Afterward remember to clean up the environment if you don't want it anymore:
```
conda deactivate
```

## Python (CuPy)

As above, set up an environment:
```
module load conda/tensorflow
conda activate
conda create -n cupy_env
conda activate cupy_env
conda install cupy
```

Now we can run and profile the application:
```
nsys profile --stats=true python daxpy_cupy.py
```

Afterward remember to clean up the environment if you don't want it anymore:
```
conda deactivate
```
