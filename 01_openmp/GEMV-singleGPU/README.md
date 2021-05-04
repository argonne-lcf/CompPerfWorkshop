Last update 2021 May 3rd

Require only 1 GPU, use `single-gpu` queue
```
qsub -A Comp_Perf_Workshop -n 1 -t 60 -I -q single-gpu
```

Need to load NVIDIA HPC SDK module on ThetaGPU
```
module load nvhpc-nompi/21.3
```

Case 1: one matrix-vector multiplication running with CPU threads.

Case 2: offload one matrix-vector multiplication running to a single GPU

```
$ nsys nvprof 2-gemv-omp-target-reduction.f.x
...
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum            Name          
 -------  ---------------  ---------  -----------  ---------  ---------  -----------------------
   100.0        1,024,766          1  1,024,766.0  1,024,766  1,024,766  nvkernel_gemv__F1L58_1_
```

Case 3: offload one matrix-vector multiplication running to a single GPU. Data movement is separated from compute kernel.
