# NVIDIA software tutorial

To get started, on a ThetaGPU allocation, do

```
$ module load nvhpc/21.3
```

## DAXPY Examples

Now we can compile and run (with profiling) each of the example codes, which
each perform a DAXPY (double precision a * x + y) in either Fortran or C++:

### Standard Language Parallelism

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
### OpenACC

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

### OpenMP

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

### cuBLAS

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

### Python (Numba)

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

### Python (CuPy)

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

## Performance Analysis Walkthrough

Let's walk through a set of steps involved with porting a CPU-only Jacobi iteration example
to run on a GPU. Our goal is to obtain "speed of light" for our application: the fastest
possible performance we can achieve (or at least as close as is practical). In an accelerated
computing environment, the GPU(s) you have access to have much higher throughput than the CPU(s).
So, if your application can take advantage of parallel processing, the approach to achieving
speed of light is first to have the GPU portion of the computation consume as large a fraction
of the application wall time as possible, and second to make that portion of the code run as
fast as possible. That means we need to remove as many performance limiters as possible.

Common limiters of the first kind are:

- Portions of the code that run serially on the CPU
- Memory allocation and copies
- Latency of launching GPU kernels

Common limiters of the second kind are:

- Not enough work to hide instruction latency
- Uncoalesced memory accesses, lack of cache reuse, not using shared memory
- Low arithmetic intensity (operations computed per byte accessed from memory

The serial code is in `jacobi.cpp`. At each step in the optimization process the goal is to
stop and reflect, using our developer tools to aid our understanding of the performance. First,
verify that it compiles and runs on the CPU, noting the output (we periodically print the error).
Remember to `module load nvhpc` if it's not already loaded on ThetaGPU.

```
$ nvc++ -o jacobi jacobi.cpp
$ ./jacobi
```

### Step 1: Add NVTX Annotations

Before we jump into GPU porting, let's first identify where most of the time is being spent in our application.
We're doing an extremely simple calculation, so you likely have a good guess, but our philosophy here will be
to measure performance carefully rather than assume we know how our application is porting (often, the performance
bottlenecks in your code will be surprising!). If you follow this methodical approach in your own application porting,
you will likely succeed.

We could use standard CPU wall timers to profile our code, but instead we will choose to use a method that has better
integration with the NVIDIA profiling tools: the NVIDIA Tools Extension, or NVTX for short. NVTX is an instrumentation
API that can be utilized by the NVIDIA developer tools to indicate sections of code with human-readable string labels.
Those labels can then be used in profiling output to demonstrate where time was spent.

The simplest NVTX API to use is the pair `nvtxRangePush()` and `nvtxRangePop()`. Using these two functions around a
region of code looks like:

```
nvtxRangePush("my region name");
// do work here
nvtxRangePop();
```

Then the string "my region name" would be used in the profiling tool to time the section of code in between the push and
the pop. The only requirement for usage is that we include the right header (`nvToolsExt.h`) and link against the right
runtime library (`libnvToolsExt.so`). Let's take a look at that in `jacobi_step1.cpp`. We'll inspect the summary output
from our profiling tool Nsight Systems. Its `--stats=true` summary output includes information about time spent in the
various NVTX regions. Note also that we're going to compile with nvcc; we're going to add CUDA code later, so we're getting
familiar with the compilation mechanics now. Note that nvcc provided with the HPC SDK uses nvc++ for compiling CPU code.

```
$ nvcc -o jacobi_step1 -x cu -lnvToolsExt jacobi_step1.cpp
$ nsys profile --stats=true -o jacobi_step1 -f true ./jacobi_step1
```

### Step 2: Unified Memory

Rather than using standard CPU malloc(), we use cudaMallocManaged() to allocate our data in Unified Memory (but we don't
make any other changes yet). This is completely legal even if, as in this case, we only intend to use host code (for now).
We also remember to use CUDA error checking. What does the profile indicate about the relative cost of starting up a CUDA program?

```
$ nvcc -o jacobi_step2 -x cu -lnvToolsExt jacobi_step2.cpp
$ nsys profile --stats=true -o jacobi_step2 -f true ./jacobi_step2
```

### Step 3: Make the Problem Bigger

In step 2, we saw that the cost of initializing CUDA (called "context creation") can be high, often measured in the hundreds
of milliseconds. In this case, the cost is so large that it dwarfs the time spent in the actual calculation. Even if we could
make the cost of the calculation zero with infinitely fast kernels, we would only make a small dent in the runtime of the
application, and it would still be much slower than the original CPU-only calculation (without Unified Memory) was. There is
simply no sense trying to optimize this scenario.

When faced with this problem, the main conclusion to reach is that you need to solve a bigger problem. In many scientific
applications there are two primary ways to make the problem bigger: we can either add more elements/zones/particles, or we
can increase the number of iterations the code runs. In this specific case, the options are to increase the number of points
in the grid, or to use a stricter error tolerance (which should require more iterations to achieve). However, if you make the
tolerance several orders of magnitude tighter, you will only increase the number of steps by a relatively small factor for this
particular case. (Most of the work is in transforming from our terrible initial guess of all zeros to a state that approximates
the correct solution; the rest is fine-tuning.) So we have to use more grid points, which will achieve a finer spatial resolution
and thus a more accurate (but expensive) answer. **This is a general fact of life when using GPUs: often it only makes sense to
solve a much more expensive problem than the one we were solving before on CPUs.**

Let's increase the number of grid points, *N*, such that the time spent in the main relaxation phase is at least 95% of the total
application time.

```
$ nvcc -o jacobi_step3 -x cu -lnvToolsExt jacobi_step3.cpp
$ nsys profile --stats=true -o jacobi_step3 -f true ./jacobi_step3
```

### Step 4: Convert the Jacobi Step to CUDA

The Jacobi relaxation steps are now the most expensive portion of the application. We also know this is a task we can solve in
parallel since each zone is updated independently (with the exception of the error, for which we must perform some sort of reduction).
Let's convert `jacobi_step()` (only) to a kernel. We parallelize over both the inner and outer loop using a two-dimensional
threadblock of size (32x32), so that the body of the function doesn't contain any loops, just the update to `f` and the error.
How much faster does the Jacobi step get? How much faster does the application get overall? What is the new application bottleneck?

```
$ nvcc -o jacobi_step4 -x cu -lnvToolsExt jacobi_step4.cpp
$ nsys profile --stats=true -o jacobi_step4 -f true ./jacobi_step4
```

### Step 5: Convert the Swap Kernel

We saw above that the Jacobi kernel got significantly faster in absolute terms, though perhaps not as much as we would have hoped,
but it made the swap kernel slower! It appears that we're paying both for the cost of Unified Memory transfers from host to device
in the Jacobi kernel, and Unified Memory transfers from device to host when the swap function occurs.

At this point we have two options to improve performance. We could either use Unified Memory prefetches to move the data more
efficiently, or we could just go ahead and port the swap kernel to CUDA as well. You're welcome to try doing the former to practice,
but we are going to suggest the latter. Our goal should be to do as much of the compute work as possible on the GPU, and in this case
it's entirely possible to keep the data on the GPU for the entirety of the Jacobi iteration.

Let's implement `swap_data()` in CUDA and check the profiler output to understand what happened.

```
$ nvcc -o jacobi_step5 -x cu -lnvToolsExt jacobi_step5.cpp
$ nsys profile --stats=true -o jacobi_step5 -f true ./jacobi_step5
```

### Step 6: Analyze the Jacobi Kernel

We are now much faster on the GPU than on the CPU, and from our profiling output we should now be able to identify that the vast
majority of the application runtime is spent in kernels, particularly the Jacobi kernel (the swap kernel apears to be very fast
in comparison). It is now appropriate to start analyzing the performance of this kernel and ask if there are any optimizations
we can apply.

First, let us hypothesize about whether an ideal implementation of this kernel should be compute-bound, memory-bound, or neither
(latency-bound). To avoid being latency-bound, we should generally expose enough work to keep a large number of threads running
on the device. If *N* = 2048, say, then there are 2048 * 2048 or about 4 million degrees of freedom in this problem. Since the
order of magnitude number of threads a modern high-end GPU can simultaneously field is O(100k), we likely do enough work to keep
the device busy -- though we will have to verify this.

When thinking about whether we are compute-bound or memory-bound, it is natural to think in terms of the arithmetic intensity
of the operation, that is, the number of (floating-point) operations computed per byte moved. A modern accelerator typically is
only compute bound when this ratio is of order 10. But the Jacobi stencil involves moving four words (each of which is 4 bytes,
for single-precision floating point) while only computing four floating point operations (three adds and one multiply). So the
arithmetic intensity is 4 (FLOPs) / 16 (bytes) = 0.25 FLOPs / byte. Clearly this is in the memory-bandwidth bound regime.

With that analysis in mind, let's see what the profiling tool tells us about the memory throughput compared to speed-of-light.
We'll apply Nsight Compute to the program we compiled in Step 5. We assume that every invocation of the kernel has approximately
similar performance characteristics, so we only profile one invocation, and we skip the first few to allow the device to warm up.
We'll save the input to a file first, and then import the results to display in the terminal (in case we want to open the report
in the Nsight Compute user interface).

```
$ ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step5 --force-overwrite ./jacobi_step5
$ ncu --import jacobi_step5.ncu-rep
```

You likely saw that, as predicted, we have pretty decent achieved occupancy, so we're giving the GPU enough work to do, but we are
definitely not close to speed-of-light on compute throughput (SM %), and disappointingly we're also not even close on memory throughput
either. We appear to still be latency-bound in some way.

A likely culprit explaining low memory throughput is poor memory access patterns. Since we're currently working with global memory, that
usually implies uncoalesced memory accesses. Could that have anything to do with it? If that's where we're headed, then at this point an
obvious question to ask is, does our threading strategy actually map well to the memory layout of our arrays? This can be a tricky thing
to sort out when working with 2D data and 2D blocks, and it gets even more complicated for the Jacobi kernel in particular because of the
stencil pattern. However, we fortunately have a kernel that we know uses exactly the same threading strategy (swap_data()) that is much
faster. What does Nsight Compute say about the memory throughput of that kernel? We can use the `--set full` option to do a more thorough
(albeit more expensive) analysis. At the bottom of the output, Nsight Compute will tell us if there were a significant amount of uncoalesced
accesses.

```
$ ncu --launch-count 1 --launch-skip 5 --kernel-regex swap_data --set full ./jacobi_step5
```

OK, so we can clearly conclude two things based on this output. First, there are plenty of uncoalesced accesses in this kernel -- in fact,
Nsight Compute tells us that we did 8x as many sector loads as we needed! This is a smoking gun for a memory access pattern with a large
stride. Correspondingly, we only got a small fraction of DRAM throughput. Second, despite this fact, we achieved a pretty decent fraction
of speed-of-light for *L2 cache* accesses. This makes sense to the extent that caching is helping to ameliorate the poor nature of our DRAM
access pattern, but a question to ask yourself is, why didn't the Jacobi kernel achieve that? We'll come back to that later.

Compare the indexing scheme to the threading scheme, noting that in a two-dimensional threadblock, the `x` dimension is the contiguous dimension
and the `y` dimension is the strided dimension; in a 32x32 thread block, you can think of `threadIdx.y` as enumerating which of 32 warps we're
using, while each warp constitutes the 32 threads in the `x` dimension. Let's fix the issue so we achieve coalesced accesses.

As a side note, we expect this to improve GPU performance, but it's also likely this would have improved the CPU performance as well, which
will make the overall GPU speedup relative to the CPU less impressive. You may want to go back and check how much of a factor that was in
the CPU-only code.

```
$ nvcc -o jacobi_step6 -x cu -lnvToolsExt jacobi_step6.cpp
$ nsys profile --stats=true -o jacobi_step6 -f true ./jacobi_step6
```

Verify using Nsight Compute that the DRAM throughput of the swap kernel is much better now.

```
$ ncu --launch-count 1 --launch-skip 5 --kernel-regex swap_data --set full ./jacobi_step6
```

###  Step 7: Revisiting the Reduction

Let's take another look at the Jacobi kernel now that we've fixed the overall global memory access pattern.

```
$ ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step6 --force-overwrite --set full ./jacobi_step6
$ ncu --import jacobi_step6.ncu-rep
```

The output from Nsight Compute is a little puzzling here. We know we probably have some work to do with our stencil operation, but surely
we should be doing better than ~1% of peak memory throughput? If we consult the Scheduler Statistics section, we see that a shocking 99.9%
of cycles have no warps eligible to issue work! This is much, much worse than the swap kernel, which has eligible warps on average 15% of
the time. What could explain that? Well, the only thing we're really doing in our kernel besides the stencil update is the atomic update to
the error counter. And we have reason to believe that if many threads are writing atomically to the same location at the same time, they
will serialize and stall. In the beginning, we suggested doing it this way because just getting the work on the GPU to begin with was the
clear first step, but now we obviously need to revisit this.

Let's refactor the kernel to use a more efficient reduction scheme that uses fewer overall atomics. To get a sense of how good of a job
you would like to do, try commenting out the atomic reduction entirely from the kernel (and then temporarily modifying `main()` so that you
can run enough iterations of the Jacobi kernel to get a profiling result, and see how much faster it is in that case (and inspect the Nsight
Compute output for that case). That's your "speed of light" kernel, at least with respect to the reduction phase.

```
$ nvcc -o jacobi_step7 -x cu -lnvToolsExt jacobi_step7.cpp
$ nsys profile --stats=true -o jacobi_step7 -f true ./jacobi_step7
```

### Step 8: Shared Memory

Similar to the thought experiment we did about how fast our reduction ought to be, we can also do a thought experiment about what the speed
of light for the Jacobi kernel is. We have a nice comparison kernel in `swap_data()`, which has fully coalesced accesses. Since there is a
significant delta in DRAM throughput between that kernel and the Jacobi kernel, we'd like to see if there's anything we can do.

Stencil operations are hard for GPU caches to deal with properly because the very large stride between row `j` and row `j+1` (corresponding to
the number of columns) means that we may get little cache reuse, a problem that only grows as the array size becomes larger. This is a good
candidate for caching the stencil data in shared memory, so that when we do reuse the data, it's reading from a higher bandwidth, lower latency
source close to the compute cores. Note that we're not trying to improve DRAM throughput, we're trying to access DRAM less frequently. Let's
implement shared memory usage in the kernel using a 2D tile. For simplicity, we assume that the number of threads in the block is 1024 and that
there are 32 per dimension (we can't get larger than this anyway), so that we can hardcode the size of the shared memory array at compile time.
There's a couple ways to do this; the simplest way is simply to read in the data into a 2D tile whose extent is 34x34 (since we need to update
32x32 values and the stencil depends on data up to one element away in each dimension). We also need to make sure we perform a proper threadblock
synchronization before reading the data in shared memory, and we note that the `__syncthreads()` intrinsic needs to be called by all threads in
the block. Note that the solution presented here is probably not the best possible solution, it errs slightly on the side of legibility over
performance in how the shared memory tile is loaded.

```
$ nvcc -o jacobi_step8 -x cu -lnvToolsExt jacobi_step8.cpp
$ nsys profile --stats=true -o jacobi_step8 -f true ./jacobi_step8
```

```
$ ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step8 --force-overwrite --set full ./jacobi_step8
$ ncu --import jacobi_step8.ncu-rep
```

### Closing Thoughts

After all of these steps, the kernels are now so fast again that the device warmup may be again a salient performance factor. In this case, we may
want to again consider increasing the size of the problem to amortize this cost out. If you do, try comparing it to the CPU implementation to see
what our final speedup was.
