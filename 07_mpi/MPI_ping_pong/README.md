# MPI Ping Pong to Demonstrate CUDA-Aware MPI

In this tutorial, we will look at a simple ping pong code that measures bandwidth for data transfers between 2 MPI ranks. We will look at a CPU-only version, a CUDA version that stages data through CPU memory, and a CUDA-Aware version that passes data directly between GPUs (using GPUDirect).

**NOTE:** This code is not optimized to achieve the best bandwidth results. It is simply meant to demonstrate how to use CUDA-Aware MPI.

## CPU Version

We will begin by looking at a CPU-only version of the code in order to understand the idea behind an MPI ping pong program. Basically, 2 MPI ranks pass data back and forth and the bandwidth is calculated by timing the data transfers and knowing the size of the data being transferred.

Let's look at the `cpu/ping_pong.c` code to see how this is implemented. At the top of the `main` program, we initialize MPI, determine the total number of MPI ranks, determine each rank's ID, and make sure we only have 2 total ranks:

``` c
    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status stat;

    if(size != 2){
        if(rank == 0){
            printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
        }
        MPI_Finalize();
        exit(0);
    }
```

Next, we enter our main `for` loop, where each iteration of the loop performs data transfers and bandwidth calculations for a different message size, ranging from 8 B to 1 GB (note that each element of the array is a double-precision variable of size 8 B, and `1 << i` can be read as "2 raised to the i power"):

``` c
    /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    for(int i=0; i<=27; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        
        ...
```

We then initialize the array `A`, set some tags to match MPI Send/Receive pairs, set `loop_count` (used later), and run a warm-up loop 5 times to remove any MPI setup costs:

``` c
        // Initialize all elements of A to 0.0
        for(int i=0; i<N; i++){
            A[i] = 0.0;
        }

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;

        // Warm-up loop
        for(int i=1; i<=5; i++){
            if(rank == 0){
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == 1){
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }
```

If you are not familiar with MPI, the MPI calls in the warm-up above loop might be a bit confusing, so an explanation here might be helpful. Essentially, each iteration of the warm-up loop is doing the following:

* If you are MPI rank 0, first send a message (the data in your array `A`) to MPI rank 1, and then expect to receive a message back from MPI rank 1 (the data in MPI rank 1's copy of array `A`). 

* If you are MPI rank 1, first expect to receive a message from rank 0 (the data in MPI rank 0's copy of array `A`), and then send a message back to MPI rank 0 (the data in your copy of array `A`).

The two bullet points above describe one "ping pong" data transfer between the MPI ranks (although these were just part of the warm-up loop). 

Getting back to the code, now we actually perform the ping-pong send and receive pairs `loop_count` times while timing the execution:

```c
        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for(int i=1; i<=loop_count; i++){
            if(rank == 0){
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == 1){
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;
```

Then, from the timing results and the known size of the data transfers, we calculate the bandwidth and print the results:

```c
        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
```

And, of course, we must free the allocated memory, finalize MPI, and exit the program:

```c
        free(A);
    }

    MPI_Finalize();

    return 0;
}
```
### Results

Running this code on Summit yields the following results:

```c
Transfer size (B):          8, Transfer Time (s):     0.000000610, Bandwidth (GB/s):     0.012215068
Transfer size (B):         16, Transfer Time (s):     0.000000585, Bandwidth (GB/s):     0.025489063
Transfer size (B):         32, Transfer Time (s):     0.000000588, Bandwidth (GB/s):     0.050648905
Transfer size (B):         64, Transfer Time (s):     0.000000660, Bandwidth (GB/s):     0.090371685
Transfer size (B):        128, Transfer Time (s):     0.000001578, Bandwidth (GB/s):     0.075540714
Transfer size (B):        256, Transfer Time (s):     0.000000894, Bandwidth (GB/s):     0.266541358
Transfer size (B):        512, Transfer Time (s):     0.000000914, Bandwidth (GB/s):     0.521914953
Transfer size (B):       1024, Transfer Time (s):     0.000001275, Bandwidth (GB/s):     0.748208720
Transfer size (B):       2048, Transfer Time (s):     0.000001691, Bandwidth (GB/s):     1.127674490
Transfer size (B):       4096, Transfer Time (s):     0.000002343, Bandwidth (GB/s):     1.627895886
Transfer size (B):       8192, Transfer Time (s):     0.000002978, Bandwidth (GB/s):     2.561893108
Transfer size (B):      16384, Transfer Time (s):     0.000005093, Bandwidth (GB/s):     2.996255182
Transfer size (B):      32768, Transfer Time (s):     0.000008885, Bandwidth (GB/s):     3.434888737
Transfer size (B):      65536, Transfer Time (s):     0.000017002, Bandwidth (GB/s):     3.589874636
Transfer size (B):     131072, Transfer Time (s):     0.000032039, Bandwidth (GB/s):     3.810059709
Transfer size (B):     262144, Transfer Time (s):     0.000021980, Bandwidth (GB/s):    11.107535215
Transfer size (B):     524288, Transfer Time (s):     0.000032818, Bandwidth (GB/s):    14.878630748
Transfer size (B):    1048576, Transfer Time (s):     0.000062507, Bandwidth (GB/s):    15.623167715
Transfer size (B):    2097152, Transfer Time (s):     0.000102356, Bandwidth (GB/s):    19.081661255
Transfer size (B):    4194304, Transfer Time (s):     0.000197997, Bandwidth (GB/s):    19.728786447
Transfer size (B):    8388608, Transfer Time (s):     0.000394892, Bandwidth (GB/s):    19.783901301
Transfer size (B):   16777216, Transfer Time (s):     0.000793706, Bandwidth (GB/s):    19.686121208
Transfer size (B):   33554432, Transfer Time (s):     0.001564052, Bandwidth (GB/s):    19.980148875
Transfer size (B):   67108864, Transfer Time (s):     0.003132349, Bandwidth (GB/s):    19.953074009
Transfer size (B):  134217728, Transfer Time (s):     0.006278125, Bandwidth (GB/s):    19.910404613
Transfer size (B):  268435456, Transfer Time (s):     0.010468920, Bandwidth (GB/s):    23.880210004
Transfer size (B):  536870912, Transfer Time (s):     0.020949296, Bandwidth (GB/s):    23.867149998
Transfer size (B): 1073741824, Transfer Time (s):     0.041904565, Bandwidth (GB/s):    23.863748251
```

## CUDA Staged Version

Now that we are familiar with a basic MPI ping pong code, let's look at a version that includes GPUs...

In this example, we still pass data back and forth between two MPI ranks, but this time the data lives in GPU memory. More specifically, MPI rank 0 has a memory buffer in GPU 0's memory and MPI rank 1 has a memory buffer in GPU 1's memory, and they will pass data back and forth between the two GPUs' memories. Here, to get data from GPU 0's memory to GPU 1's memory, we will first stage the data through CPU memory. 

Now, let's take a look at the code to see the differences from the CPU-only version. Before `main`, we define a macro that allows us to check for errors in our CUDA API calls. This isn't important for this tutorial but, in general, it's a good idea to include such error checks.

```c
// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)
```

Similar to the CPU-only version, just inside `main`, we initialize MPI and find each MPI rank's ID, but here we also map the MPI rank to a different GPU (i.e., MPI rank 0 is mapped to GPU 0 and MPI rank 1 is mapped to GPU 1). Notice that we have wrapped the `cudaSetDevice()` call in our `cudaErrorCheck` macro (again, this isn't necessary, just good practice in CUDA programs).

```c
    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status stat;

    if(size != 2){
        if(rank == 0){
            printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
        }
        MPI_Finalize();
        exit(0);
    }

    // Map MPI ranks to GPUs
    cudaErrorCheck( cudaSetDevice(rank) );
```
Next, we do roughly the same as we did in the CPU-only version: enter our main `for` loop that iterates over the different message sizes, allocate and intialize array `A`, and run our warm-up loop. However, we now have a call to `cudaMalloc` to allocate a memory buffer (`d_A`) on the GPUs and a call to `cudaMemcpy` to transfer the data initialized in array `A` to the GPU array (buffer) `d_A`. The `cudaMemcpy` was needed to get the data to the GPU before starting our ping pong. 

There are also `cudaMemcpy` calls within the if statements of the warm-up loop. These are needed to transfer data from the GPU buffer to the CPU buffer before the CPU buffer is used in the MPI call (and similarly for the transfer back).

```c
    /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    for(int i=0; i<=27; i++){

        long int N = 1 << i;
   
        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));

        // Initialize all elements of A to 0.0
        for(int i=0; i<N; i++){
            A[i] = 0.0;
        }

        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;

        // Warm-up loop
        for(int i=1; i<=5; i++){
            if(rank == 0){
                cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
                cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            }
            else if(rank == 1){
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
                cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }
```

To clarify this, let's look at the following diagram of a Summit node. The magenta line shows the path taken by data as its passed from GPU 0's memory to GPU 1's memory (assuming the top left device is GPU 0 and the one below that is GPU 1). There are 3 steps involved:

* Data must first be transferred from GPU 0's memory into CPU memory
* Then an MPI call is used to pass the data from MPI rank 0 to MPI rank 1 (in CPU memory)
* Now that MPI rank 1 has the data (in CPU memory), it can transfer the data to GPU 1's memory

Or, more explicitly:

* MPI rank 0 must first transfer the data from a buffer in GPU 0's memory into a buffer in CPU memory
* Then, MPI rank 0 can use its CPU buffer to send data to MPI rank 1's CPU buffer
* Now that MPI rank 1 has the data in its CPU memory buffer, it can transfer it to a buffer in GPU 1's memory.


Getting back to the code, we now perform our actual ping pong loop (with the same structure as the warm-up loop we just discussed) while timing the execution:

```c
        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();
   
        for(int i=1; i<=loop_count; i++){
            if(rank == 0){
                cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
                MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
                cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            }
            else if(rank == 1){
                MPI_Recv(A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
                cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
                MPI_Send(A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;
```

Similar to the CPU-only case, from the timing results and the known size of the data transfers, we calculate the bandwidth and print the results:

```c
        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
```

And finally, we free the memory on both the CPU and GPU, finalize MPI, and exit the program.

```c
        cudaErrorCheck( cudaFree(d_A) );
        free(A);
    }

    MPI_Finalize();

    return 0;
}
```

### Results

Running this code yields the results below. The lower bandwidth obtained in this example is not surprising considering the data transfers between CPU and GPU that are not present in the CPU only version. 

```c
Transfer size (B):          8, Transfer Time (s):     0.000020403, Bandwidth (GB/s):     0.000365172
Transfer size (B):         16, Transfer Time (s):     0.000020564, Bandwidth (GB/s):     0.000724618
Transfer size (B):         32, Transfer Time (s):     0.000020956, Bandwidth (GB/s):     0.001422118
Transfer size (B):         64, Transfer Time (s):     0.000020820, Bandwidth (GB/s):     0.002862863
Transfer size (B):        128, Transfer Time (s):     0.000020755, Bandwidth (GB/s):     0.005743664
Transfer size (B):        256, Transfer Time (s):     0.000020953, Bandwidth (GB/s):     0.011378846
Transfer size (B):        512, Transfer Time (s):     0.000021163, Bandwidth (GB/s):     0.022531654
Transfer size (B):       1024, Transfer Time (s):     0.000021400, Bandwidth (GB/s):     0.044563929
Transfer size (B):       2048, Transfer Time (s):     0.000021855, Bandwidth (GB/s):     0.087271675
Transfer size (B):       4096, Transfer Time (s):     0.000024189, Bandwidth (GB/s):     0.157703802
Transfer size (B):       8192, Transfer Time (s):     0.000026381, Bandwidth (GB/s):     0.289197613
Transfer size (B):      16384, Transfer Time (s):     0.000031257, Bandwidth (GB/s):     0.488169401
Transfer size (B):      32768, Transfer Time (s):     0.000046707, Bandwidth (GB/s):     0.653378357
Transfer size (B):      65536, Transfer Time (s):     0.000058954, Bandwidth (GB/s):     1.035306802
Transfer size (B):     131072, Transfer Time (s):     0.000087381, Bandwidth (GB/s):     1.396982636
Transfer size (B):     262144, Transfer Time (s):     0.000140982, Bandwidth (GB/s):     1.731718672
Transfer size (B):     524288, Transfer Time (s):     0.000266125, Bandwidth (GB/s):     1.834779795
Transfer size (B):    1048576, Transfer Time (s):     0.000484124, Bandwidth (GB/s):     2.017175236
Transfer size (B):    2097152, Transfer Time (s):     0.000803233, Bandwidth (GB/s):     2.431579144
Transfer size (B):    4194304, Transfer Time (s):     0.001656954, Bandwidth (GB/s):     2.357488628
Transfer size (B):    8388608, Transfer Time (s):     0.003207830, Bandwidth (GB/s):     2.435447327
Transfer size (B):   16777216, Transfer Time (s):     0.006903458, Bandwidth (GB/s):     2.263358319
Transfer size (B):   33554432, Transfer Time (s):     0.015314410, Bandwidth (GB/s):     2.040561734
Transfer size (B):   67108864, Transfer Time (s):     0.029280032, Bandwidth (GB/s):     2.134560485
Transfer size (B):  134217728, Transfer Time (s):     0.058107654, Bandwidth (GB/s):     2.151179596
Transfer size (B):  268435456, Transfer Time (s):     0.118616563, Bandwidth (GB/s):     2.107631467
Transfer size (B):  536870912, Transfer Time (s):     0.238929797, Bandwidth (GB/s):     2.092664898
Transfer size (B): 1073741824, Transfer Time (s):     0.469888118, Bandwidth (GB/s):     2.128166176
```

## CUDA-Aware Version

Before looking at this code example, let's first describe CUDA-Aware MPI and GPUDirect. These two topics are often used interchangeably, and although they can be related, they are distinct topics. 

CUDA-Aware MPI is an MPI implementation that allows GPU buffers (e.g., GPU memory allocated with `cudaMalloc`) to be used directly in MPI calls. However, CUDA-Aware MPI by itself does not specify whether data is staged through CPU memory or passed directly from GPU to GPU. That's where GPUDirect comes in! 

GPUDirect can enhance CUDA-Aware MPI by allowing data transfers directly between GPUs on the same node (peer-to-peer) or directly between GPUs on different nodes (RDMA support) without the need to stage data through CPU memory. 

Now let's take a look at the code. It's essentially the same as the CUDA staged version but now there are no calls to `cudaMemcpy` during the ping pong steps. Instead, we use our GPU buffers (`d_A`) directly in the MPI calls:

```c
        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for(int i=1; i<=loop_count; i++){
            if(rank == 0){
                MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == 1){
                MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;
```

### Results

There is a noticeable improvement in bandwidth relative to the CUDA staged version. This is because Summit has GPUDirect support (both peer-to-peer and RDMA) that allows data transfers between peer GPUs across NVLink.

```c
Transfer size (B):          8, Transfer Time (s):     0.000015785, Bandwidth (GB/s):     0.000472016
Transfer size (B):         16, Transfer Time (s):     0.000015637, Bandwidth (GB/s):     0.000952966
Transfer size (B):         32, Transfer Time (s):     0.000015594, Bandwidth (GB/s):     0.001911085
Transfer size (B):         64, Transfer Time (s):     0.000015657, Bandwidth (GB/s):     0.003806789
Transfer size (B):        128, Transfer Time (s):     0.000015745, Bandwidth (GB/s):     0.007571041
Transfer size (B):        256, Transfer Time (s):     0.000015763, Bandwidth (GB/s):     0.015125338
Transfer size (B):        512, Transfer Time (s):     0.000015635, Bandwidth (GB/s):     0.030497240
Transfer size (B):       1024, Transfer Time (s):     0.000015797, Bandwidth (GB/s):     0.060372011
Transfer size (B):       2048, Transfer Time (s):     0.000016035, Bandwidth (GB/s):     0.118947011
Transfer size (B):       4096, Transfer Time (s):     0.000016338, Bandwidth (GB/s):     0.233479753
Transfer size (B):       8192, Transfer Time (s):     0.000016270, Bandwidth (GB/s):     0.468926370
Transfer size (B):      16384, Transfer Time (s):     0.000016056, Bandwidth (GB/s):     0.950335668
Transfer size (B):      32768, Transfer Time (s):     0.000016178, Bandwidth (GB/s):     1.886322037
Transfer size (B):      65536, Transfer Time (s):     0.000016172, Bandwidth (GB/s):     3.774118416
Transfer size (B):     131072, Transfer Time (s):     0.000016766, Bandwidth (GB/s):     7.280842403
Transfer size (B):     262144, Transfer Time (s):     0.000017187, Bandwidth (GB/s):    14.204834717
Transfer size (B):     524288, Transfer Time (s):     0.000018376, Bandwidth (GB/s):    26.571379517
Transfer size (B):    1048576, Transfer Time (s):     0.000020024, Bandwidth (GB/s):    48.769528412
Transfer size (B):    2097152, Transfer Time (s):     0.000024074, Bandwidth (GB/s):    81.131068344
Transfer size (B):    4194304, Transfer Time (s):     0.000031821, Bandwidth (GB/s):   122.757061536
Transfer size (B):    8388608, Transfer Time (s):     0.000045985, Bandwidth (GB/s):   169.893058164
Transfer size (B):   16777216, Transfer Time (s):     0.000076010, Bandwidth (GB/s):   205.565922656
Transfer size (B):   33554432, Transfer Time (s):     0.000136284, Bandwidth (GB/s):   229.300140750
Transfer size (B):   67108864, Transfer Time (s):     0.000256476, Bandwidth (GB/s):   243.687670542
Transfer size (B):  134217728, Transfer Time (s):     0.000494715, Bandwidth (GB/s):   252.670607034
Transfer size (B):  268435456, Transfer Time (s):     0.000971699, Bandwidth (GB/s):   257.281400678
Transfer size (B):  536870912, Transfer Time (s):     0.001928221, Bandwidth (GB/s):   259.306330465
Transfer size (B): 1073741824, Transfer Time (s):     0.003841230, Bandwidth (GB/s):   260.333286796
```

The magenta line in the following diagram shows how the data is transferred between the two GPUs across NVLink, which is the reason for the improved performance. 


### Additional Notes

