# Pi By Parsl Tutorial

The files in this directory demonstrate various iterations on the Parsl workflow tool
as run on the Theta resource of ALCF. Using the basic example of pi as computed via
Monte Carlo methods, we outline the major features of Parsl --- particularly the use
of both python and bash "apps" to launch tasks, the capability of distributing work
both locally and across compute nodes, and the usage of monitoring tools to create
databases for optimization analysis.

## Installation

Installing parsl on Theta is a straightforward process.

1. Load a miniconda environment. For this demo, we use the following:
`module load miniconda-3`

2. Install parsl locally via pip:
 `pip install --user parsl`

3. Optionally, install additional monitoring packages:
`pip install --user parsl[monitoring]`

## Running

The several codes in this respository are designed to demonstrate the usage of parsl in
several key aspects. In each case, pi is calculated by generating random data points,
determining how many lie within a quarter circle, and taking the ratio of those areas
and multiplying by four to get an estimate of pi. The more points, the more accurate this
method is.

In each case, pay attention to the following command:
    parsl_config = Config()
    parsl.load(parsl_config)
These two commands define which executors Parsl will use to run commands and modifies the
configuration of them appropriately. We speak more about them in detail below.

### ThreadPoolExecutor, Python
For an entirely local execution using Python, please look at pi\_test\_local.py.

The first executor to look at is the ThreadPoolExecutor. This runs commands using local
threads---in this case running on the Theta log-in node. For best practices, we encourage
only using this executor for commands that are not resource intensive. The Parsl executor
here is defined by having a number of threads to utilize.

The estimate\_pi function calculates the value of the pi by the method described above. Note
that the numpy library should be imported within the function call, as each Parsl worker will
be running this command independently of the driver code. The @python\_app wrapper informs
Parsl that this python function should be sent to Parsl workers---the optional executors keyword
tells it which labeled executor to use (we will explore this again later).

The for loop over estimate\_pi then requests this task to be done by the parsl workers for many
times over. Getting the result requires an additional call of .result(), as seen in the below list
comprehension.

We then simply take the mean of these results to calculate pi.

This entire code can be called via:
```
python pi_test_local.py
```

### ThreadPoolExecutor, Bash
Next look at the pi\_test\_bash.py script.

This script also makes use of only local executors, although with the addition of @bash\_app commands.
Here, we have made the pi estimation a separate python script, which carries out the random number
generation alone. Commands run using @bash\_app will not return Python objects, so we store the output
of each job as a file. These files are then concatenated, to a single file and then the output is read.

Note that for @bash\_app functions, whatever is returned will be run as Bash shell code.

This entire code can be called by running:
```
python pi_test_bash.py
```

It will generate a lot of text files in this process! Feel free to remove these afterward.

### HighThroughputExecutor
Next up is the pi\_test\_queue.py file.

Rather than running everything on the local machine, this code will instead be submitting a job to
Theta via Cobalt job submissions. As such, the start of the file includes additional information to
fill out. Of note:

1. The MY\_USER\_PATH entries is necessary for exporting your local user path information to Parsl
workers. This is necessary for them to find the Parsl install. This can be avoided by doing a full clone
of the repository, if desired.
2. The MY\_ALLOCATION and MY\_QUEUE entries contain information on what queue to submit to and which
account to charge the time to.
3. The MY\_COMPUTE\_NODES and MY\_TIME entries inform the executor what resources to request for each
job submission.
4. The MY\_COMPUTE\_BLOCKS entry determines how many jobs to run simultaneously.

In the executor configuration, note that the max\_workers defines how many workers can be launched
by Parsl across all nodes. For more detailed set-up regarding worker distribution, please see the
full [Parsl documentation](parsl.readthedocs.io).

Note that this set-up will calculate the mean locally, while running all the individual trials on
the allocated compute nodes.

This can be run with:
 ```
 python pi_test_queue.py
 ```

Note that unlike previous attempts, this will take some time to complete, based on how long it takes
to get the necessary resources. We suggest using this in a screen in order to avoid interruptions.

### MonitoringHub Database
One additional benefit to Parsl is the ability to collect data regarding all processes into a sqlite
database. We demonstrate that in the `pi_test_monitoring.py` script.

This contains an additional few lines at the start: specifically, an import command for MonitoringHub,
and under the configuration of parsl, an additional keyword argument of monitoring. MonitoringHub will
create a database in the directory the Parsl driver is run in, storing information on all executors.

One utilization tool is the `parsl visualizer`, with full documentation found [here](https://parsl.readthedocs.io/en/stable/userguide/monitoring.html). 
We also provide the python script `monitoring_test.py`, which navigates through the database using
sqlite3 in order to provide a quick example of the information contained within.

In short, the major tables existing:
1. workflow: contains information regarding the workflow as a whole---durations, hosting information,
logging, and more.
2. task: contains information for each parsl task that has been launched, including dependencies,
executor used, submission time, failure count, and input/output data.
3. node: contains information about the nodes used in a run.
4. status: contains informattion regarding the status of each taskk thatt was launched at a
given timestamp.
5. resource: contains information on resource utilization during the entirety of the run.
