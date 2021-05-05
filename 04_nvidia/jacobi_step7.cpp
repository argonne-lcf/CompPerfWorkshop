#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <nvToolsExt.h>

#define N 2048

#define IDX(i, j) ((i) + (j) * N)

// error checking macro
#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)

void allocate_memory (float** f, float** f_old, float** error) {
    cudaMallocManaged(f, N * N * sizeof(float));
    cudaMallocManaged(f_old, N * N * sizeof(float));
    cudaMallocManaged(error, sizeof(float));
    cudaCheckErrors("Memory allocation");
}

void free_memory (float* f, float* f_old, float* error) {
    cudaFree(f);
    cudaFree(f_old);
    cudaFree(error);
    cudaCheckErrors("Memory deallocation");
}

void initialize_data (float* f) {
    // Set up simple sinusoidal boundary conditions
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {

            if (i == 0 || i == N-1) {
                f[IDX(i,j)] = sin(j * 2 * M_PI / (N - 1));
            }
            else if (j == 0 || j == N-1) {
                f[IDX(i,j)] = sin(i * 2 * M_PI / (N - 1));
            }
            else {
                f[IDX(i,j)] = 0.0f;
            }

        }
    }
}

__global__ void jacobi_step (float* f, float* f_old, float* error) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    float err = 0.0f;

    if (j >= 1 && j <= N-2) {
        if (i >= 1 && i <= N-2) {
            f[IDX(i,j)] = 0.25f * (f_old[IDX(i+1,j)] + f_old[IDX(i-1,j)] +
                                   f_old[IDX(i,j+1)] + f_old[IDX(i,j-1)]);

            float df = f[IDX(i,j)] - f_old[IDX(i,j)];
            err = df * df;
        }
    }

    // Sum over threads in the warp
    // For simplicity, we do this outside the above conditional
    // so that all threads participate
    for (int offset = 16; offset > 0; offset /= 2) {
        err += __shfl_down_sync(0xffffffff, err, offset);
    }

    // If we're thread 0 in the warp, update our value to shared memory
    // Note that we're assuming exactly a 32x32 block and that the warp ID
    // is equivalent to threadIdx.y. For the general case, we would have to
    // write more careful code.
    __shared__ float reduction_array[32];
    if (threadIdx.x == 0) {
        reduction_array[threadIdx.y] = err;
    }

    // Synchronize the block before reading any values from smem
    __syncthreads();

    // Using the first warp in the block, reduce over the partial sums
    // in the shared memory array.
    if (threadIdx.y == 0) {
        err = reduction_array[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            err += __shfl_down_sync(0xffffffff, err, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(error, err);
        }
    }
}

__global__ void swap_data (float* f, float* f_old) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= 1 && j <= N-2) {
        if (i >= 1 && i <= N-2) {
            f_old[IDX(i,j)] = f[IDX(i,j)];
        }
    }
}

int main () {
    // Begin wall timing
    std::clock_t start_time = std::clock();

    float* f;
    float* f_old;
    float* error;

    // Reserve space for the scalar field and the "old" copy of the data
    nvtxRangePush("Allocate memory");
    allocate_memory(&f, &f_old, &error);
    nvtxRangePop();

    // Initialize data (we'll do this on both f and f_old, so that we don't
    // have to worry about the boundary points later)
    nvtxRangePush("Initialize data");
    initialize_data(f);
    initialize_data(f_old);
    nvtxRangePop();

    // Initialize error to a large number
    *error = std::numeric_limits<float>::max();
    const float tolerance = 1.e-4f;

    // Iterate until we're converged (but set a cap on the maximum number of
    // iterations to avoid any possible hangs)
    const int max_iters = 1000;
    int num_iters = 0;

    while (*error > tolerance && num_iters < max_iters) {
        // Initialize error to zero (we'll add to it the following step)
        *error = 0.0f;

        // Perform a Jacobi relaxation step
        nvtxRangePush("Jacobi step");
        jacobi_step<<<dim3(N / 32, N / 32), dim3(32, 32)>>>(f, f_old, error);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // Swap the old data and the new data
        // We're doing this explicitly for pedagogical purposes, even though
        // in this specific application a std::swap would have been OK
        nvtxRangePush("Swap data");
        swap_data<<<dim3(N / 32, N / 32), dim3(32, 32)>>>(f, f_old);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // Normalize the L2-norm of the error by the number of data points
        // and then take the square root
        *error = std::sqrt(*error / (N * N));

        // Periodically print out the current error
        if (num_iters % 25 == 0) {
            std::cout << "Error after iteration " << num_iters << " = " << *error << std::endl;
        }

        // Increment the iteration count
        ++num_iters;
    }

    // If we took fewer than max_iters steps and the error is below the tolerance,
    // we succeeded. Otherwise, we failed.

    if (*error <= tolerance && num_iters < max_iters) {
        std::cout << "Success!" << std::endl;
    }
    else {
        std::cout << "Failure!" << std::endl;
        return -1;
    }

    // Clean up memory allocations
    nvtxRangePush("Free memory");
    free_memory(f, f_old, error);
    nvtxRangePop();

    // End wall timing
    double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
    std::cout << "Run time = " << std::setprecision(4) << duration << " seconds" << std::endl;

    return 0;
}
