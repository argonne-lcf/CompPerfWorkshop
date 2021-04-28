import parsl
import os
from parsl.app.app import bash_app, python_app
from parsl.data_provider.files import File
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

@bash_app(executors=['login-node'])
def calc_pi(num_points, outputs=[]):
    return 'python /home/antoniov/repos/parsl-pi-test/calc_pi.py {} &> {}'.format(num_points, outputs[0]) 

@bash_app(executors=['login-node'])
def concat(inputs=[], outputs=[]):
    return "cat {0} > {1}".format(" ".join(i.filepath for i in inputs), outputs[0])

@python_app(executors=['login-node'])
def total(inputs=[]):
    total = 0
    with open(inputs[0], 'r') as f:
        for l in f:
            total += float(l.strip())
    return total

if __name__ == '__main__':
    import numpy as np

    ntrials = 128
    num_points = 100000

    output_files = []
    for i in range(ntrials):
        output_files.append(calc_pi(num_points,outputs=[File(os.path.join(os.getcwd(), 'trial-%s.txt' % i))]))

    cc = concat(inputs=[i.outputs[0] for i in output_files],
                        outputs=[File(os.path.join(os.getcwd(), 'combined.txt'))])

    total = total(inputs=[cc.outputs[0]])
    pi = total.result() / ntrials
    print('pi is {}'.format(pi))
