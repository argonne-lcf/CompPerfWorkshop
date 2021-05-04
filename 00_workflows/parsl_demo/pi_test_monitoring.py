import parsl
import os
from parsl.app.app import python_app

from parsl.config import Config
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.addresses import address_by_hostname
from parsl.monitoring.monitoring import MonitoringHub

import logging

MY_USER_PATH = '/home/USERNAME/.local/miniconda-3/latest/bin/'
MY_ALLOCATION = 'Comp_Perf_Workshop'
MY_QUEUE = 'comp_perf_workshop'
MY_COMPUTE_NODES = 1
MY_COMPUTE_BLOCKS = 1
MY_TIME = '00:05:00'

WORKERS_PER_NODE = 64

parsl_config = Config(
    executors=[
        HighThroughputExecutor(
            label='theta-htex',
            max_workers = WORKERS_PER_NODE*MY_COMPUTE_NODES*MY_COMPUTE_BLOCKS,
            worker_debug=True,
            address=address_by_hostname(),
            provider=CobaltProvider(
                queue=MY_QUEUE,
                account=MY_ALLOCATION,
                launcher=AprunLauncher(overrides="-d 64"),
                walltime=MY_TIME,
                nodes_per_block=MY_COMPUTE_NODES,
                init_blocks=1,
                min_blocks=1,
                max_blocks=MY_COMPUTE_BLOCKS,
                # string to prepend to #COBALT blocks in the submit
                # script to the scheduler eg: '#COBALT -t 50'
                scheduler_options='',
                # Command to be run before starting a worker, such as:
                worker_init='module load miniconda-3; export PATH=$PATH:{}'.format(MY_USER_PATH),
                cmd_timeout=120,
            ),
        ),
        ThreadPoolExecutor(
            label='login-node',
            max_threads = 8
        ),
    ],
   monitoring=MonitoringHub(
       hub_address=address_by_hostname(),
       hub_port=55055,
       monitoring_debug=False,
       resource_monitoring_interval=10,
   )
)
parsl.load(parsl_config)

@python_app(executors=['theta-htex'])
def pi(num_points):
    from random import random

    inside = 0
    for i in range(num_points):
        x, y = random(), random()  # Drop a random point in the box.
        if x**2 + y**2 < 1:        # Count points within the circle.
            inside += 1

    return (inside*4 / num_points)

# App that computes the mean
@python_app(executors=['login-node'])
def mean(estimates):
    import numpy as np
    estimates = np.array(estimates)
    return (np.mean(estimates))

if __name__ == '__main__':
    num_points_per_trial = 10000
    num_trials = 1280
    trials = []
    for i in range(num_trials):
        trial = pi(num_points_per_trial)
        trials.append(trial)
    trials_results = [trial.result() for trial in trials]
    pi = mean(trials_results)
    print(pi.result())

