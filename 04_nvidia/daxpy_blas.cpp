#include <cstdlib>
#include <cublas_v2.h>

int main() {
    double* x;
    double* y;

    const int N = 1000000;
    const double a = 3.0;

    x = (double*) malloc(N * sizeof(double));
    y = (double*) malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        x[i] = 1.0 * i;
        y[i] = 2.0 * i;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDaxpy(handle, N, &a, x, 1, y, 1);

    cublasDestroy(handle);

    free(x);
    free(y);
}
