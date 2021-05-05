import numpy as np
import cupy as cp

N = 1000000

x = np.zeros(N, dtype=np.float64)
y = np.zeros(N, dtype=np.float64)

for i in range(N):
    x[i] = 1.0 * i
    y[i] = 2.0 * i

d_x = cp.array(x)
d_y = cp.array(y)

a = 3.0

d_y = a * d_x + d_y
