import numpy as np
from numba import vectorize

@vectorize('float64(float64, float64, float64)', target='cuda')
def daxpy(a, x, y):
    return a * x + y

N = 1000000

x = np.zeros(N, dtype=np.float64)
y = np.zeros(N, dtype=np.float64)

for i in range(N):
    x[i] = 1.0 * i
    y[i] = 2.0 * i

a = 3.0

y = daxpy(a, x, y)
