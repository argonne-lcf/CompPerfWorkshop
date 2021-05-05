#include <cstdlib>
#include <algorithm>
#include <execution>

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

    std::transform(std::execution::par_unseq, x, x + N, y, y,
                   [=] (double x_loc, double y_loc)
                   {
                       return a * x_loc + y_loc;
                   });

    free(x);
    free(y);
}
