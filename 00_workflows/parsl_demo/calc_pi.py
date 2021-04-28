from random import random
import numpy as np
import sys

def pi(num_points):
    inside = 0
    for i in range(num_points):
        x, y = random(), random()
        if x**2 + y**2 < 1:
            inside += 1

    return (inside*4 / num_points)

def mean(estimates):
    estimates = np.array(estimates)
    return (np.mean(estimates))

if __name__ == '__main__':
    num_points = int(sys.argv[1])
    pi = pi(num_points)
    print(pi)
