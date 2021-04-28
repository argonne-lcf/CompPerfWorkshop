import parsl
import os
from parsl.app.app import python_app

from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

parsl_config = Config(
    executors=[ThreadPoolExecutor(
        max_threads=8,
        label='login-node'
        )
    ],
    strategy=None,
)
parsl.load(parsl_config)

@python_app(executors=['login-node'])
def estimate_pi(n_points):
    import numpy as np
    x = np.random.uniform(0,1,n_points)
    y = np.random.uniform(0,1,n_points)
    dist = np.sqrt(x*x+y*y)
    n_circle = np.sum(dist <= 1)
    pi_est = 4*n_circle/n_points
    return pi_est

if __name__ == '__main__':
    import numpy as np
    n_points = 100000
    n_trials = 100
    trials = []
    for i in range(n_trials):
        trials.append(estimate_pi(n_points))

    outputs = [i.result() for i in trials]
    print(np.mean(outputs))
