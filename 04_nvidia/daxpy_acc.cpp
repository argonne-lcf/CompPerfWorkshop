#include <cstdlib>

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

#pragma acc parallel loop
    for (int i = 0; i < N; ++i) {
        y[i] = a * x[i] + y[i];
    }

    free(x);
    free(y);
}
