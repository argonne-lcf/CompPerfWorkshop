Balsam: ALCF Computational Performance Workshop 2021
=======================================================

To get started on ThetaGPU:

```bash
# clone this repo
git clone https://github.com/argonne-lcf/CompPerfWorkshop.git
cd CompPerfWorkshop/00_workflows/balsam_demo
```


```
# Create a virtual environment
/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python -m venv env
source env/bin/activate
python -m pip install --upgrade pip

# Install Balsam
python -m pip install balsam

# Create a Balsam site
balsam site init -n thetagpu_tutorial thetagpu_tutorial
pushd thetagpu_tutorial
balsam site start
popd
```

Create an application in Balsam (hello.py)
We define an application by wrapping the command line in a small amount of Python code. Note that the command line is directly represented, and the say_hello_to parameter will be supplied when a job uses this application.

```python
from balsam.api import ApplicationDefinition

class Hello(ApplicationDefinition):
    site = "thetagpu_tutorial"
    command_template = "echo Hello, {{ say_hello_to }}! CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
Hello.sync()
```

To add the Hello application to the Balsam site, we run the hello.py file
```bash
python hello.py
```


Create a Hello job using the Balsam CLI interface (1_create_job.sh)
```python
#!/bin/bash -x 

# Create the Hello app
python hello.py

# List apps
balsam app ls --site thetagpu_tutorial

# Create a Hello job
# Note: tag it with the key-value pair workflow=hello for easy querying later
balsam job create --site thetagpu_tutorial --app Hello --workdir=demo/hello --param say_hello_to=world --tag workflow=hello --yes

# The job resides in the Balsam server now; list the job
balsam job ls --tag workflow=hello

# Submit a batch job to run this job on ThetaGPU
# Note: the command-line parameters are similar to scheduler command lines
# Note: this job will run only jobs with a matching tag
#balsam queue submit --site thetagpu_tutorial -n 1 -t 10 -q full-node -A Comp_Perf_Workshop --tag workflow=hello -j mpi
balsam queue submit --site thetagpu_tutorial -n 1 -t 10 -q single-gpu -A datascience --tag workflow=hello -j mpi

# List the Balsam BatchJob
# Note: Balsam will submit this job to Cobalt, so it will appear in qstat output shortly
balsam queue ls 

# List status of the Hello job
balsam job ls --tag workflow=hello
```

Submit a BatchJob to run the Hello job (2_submit_batchjob.sh)
```bash
#!/bin/bash
balsam queue submit \
  -n 1 -t 10 -q full-node -A datascience \
  --site thetagpu_tutorial \
  --tag workflow=hello \
  --job-mode mpi 
```

Create a collection of jobs using the Balsam Python API (3_create_multiple_jobs.py)
```python
#!/usr/bin/env python
from balsam.api import Job
jobs = [
    Job( site_name='thetagpu_tutorial', 
         app_id="Hello", 
         workdir=f"demo/hello_multi{n}", 
         parameters={"say_hello_to": f"world {n}!"},
         tags={"workflow":"hello_multi"}, 
         node_packing_count=8, 
         gpus_per_rank=1)
    for n in range(8)
]

# Create all n jobs in one call; the list of created jobs is returned
jobs = Job.objects.bulk_create(jobs)

```

Create a collection of jobs with dependencies (4_create_multiple_jobs_with_deps.py)
```python
#!/usr/bin/env python
from balsam.api import Job,BatchJob

# Create a collection of jobs, each one depending on the job before it
n=0
job = Job( site_name='thetagpu_tutorial',
           app_id="Hello", 
           workdir=f"demo/hello_deps{n}", 
           parameters={"say_hello_to": f"world {n}!"},
           tags={"workflow":"hello_deps"}, 
           node_packing_count=8, 
           gpus_per_rank=1)
job.save()
for n in range(7):
    job = Job( site_name='thetagpu_tutorial', 
               app_id="Hello", 
               workdir=f"demo/hello_deps{n}", 
               parameters={"say_hello_to": f"world {n}!"},
               tags={"workflow":"hello_deps"}, 
               node_packing_count=8, 
               gpus_per_rank=1, 
               parent_ids=[job.id])  # Sets a dependency on the prior job
    job.save()

# Create a BatchJob to run jobs with the workflow=hello_deps tag
BatchJob.objects.create(
    site_id=287,
    num_nodes=1,
    wall_time_min=10,
    queue="single-gpu",
    project="datascience",
    job_mode="mpi",
    filter_tags={"workflow":"hello_deps"},
    #queue="full-node",
)
```

Use the Python API to monitor jobs (5_monitor_jobs.py)
```python
from datetime import datetime,timedelta
from balsam.api import EventLog

yesterday = datetime.utcnow() - timedelta(days=1)
#for evt in EventLog.objects.filter(tags={"workflow": "hello_multi"}):
for evt in EventLog.objects.filter(timestamp_after=yesterday):
    print("Job:",evt.job_id)  # Job ID
    print(evt.timestamp)      # Time of state change (UTC)
    print(evt.from_state)     # From which state the job transitioned
    print(evt.to_state)       # To which state
    print(evt.data)           # optional payload
```


The Python API includes analytics support for utilization and throughput (6_analytics.py)
```python
#!/usr/bin/env python
from balsam.api import models
from balsam.api import EventLog,Job
from balsam.analytics import throughput_report
from balsam.analytics import utilization_report
from matplotlib import pyplot as plt

# Fetch jobs and events for the Hello app
app = models.App.objects.get(site_name="thetagpu_tutorial",name="Hello")
jl = Job.objects.filter(app_id=app.id)
events = EventLog.objects.filter(job_id=[job.id for job in jl])

# Generate a throughput report
times, done_counts = throughput_report(events, to_state="JOB_FINISHED")

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
fig = plt.Figure()
plt.step(elapsed_minutes, done_counts, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Jobs completed")
plt.savefig('throughput.png')

# Generate a utilization report
times, util = utilization_report(events, node_weighting=True)

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
plt.step(elapsed_minutes, util, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Utilization")
plt.savefig("utilization.png")
```

Supplemental: Create a collection of jobs at multiple sites (7_create_jobs_at_multiple_sites.sh)
By setting dependencies between jobs, we can build up simple linear workflows, and complex workflows with multiple dependencies
Note: This example will only work for you if you replicate this site/app setup
```python
#!/bin/bash

# Create jobs at four sites
echo ThetaKNL
balsam job create --site thetaknl_tutorial --app Hello --workdir multisite/thetaknl --param say_hello_to=thetaknl --tag workflow=hello_multisite --yes

echo Cooley
balsam job create --site cooley_tutorial --app Hello --workdir multisite/cooleylogin2 --param say_hello_to=cooleylogin2 --tag workflow=hello_multisite --yes

echo ThetaGPU
balsam job create --site thetagpu_tutorial --app Hello --workdir multisite/thetagpu --param say_hello_to=thetagpu --tag workflow=hello_multisite --yes

echo Laptop
balsam job create --site tom_laptop --app Hello --workdir multisite/tom_laptop --param say_hello_to=tom_laptop --tag workflow=hello_multisite --yes

# List the jobs
balsam job ls --tag workflow=hello_multisite
```


Create a collection of jobs across Sites...
```bash
#!/bin/bash

# create jobs at four sites
balsam job create --site theta_tutorial --app Hello --workdir multisite/thetaknl --param say_hello_to=thetaknl --tag workflow=hello_multisite --yes
balsam job create --site cooley_tutorial --app Hello --workdir multisite/cooleylogin2 --param say_hello_to=cooleylogin2 --tag workflow=hello_multisite --yes
balsam job create --site 281 --app Hello --workdir multisite/thetagpu --param say_hello_to=thetagpu --tag workflow=hello_multisite --yes
balsam job create --site tom_laptop --app Hello --workdir multisite/tom_laptop --param say_hello_to=tom_laptop --tag workflow=hello_multisite --yes

# list the jobs
balsam job ls --tag workflow=hello_multisite
```
...and submit a BatchJob to run them
```bash
#!/bin/bash

# Submit BatchJobs at multiple sites
# thetagpu
# note: should use full-node queue
balsam queue submit \
  -n 1 -t 10 -q single-gpu -A datascience \
  --site thetagpu_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# theta knl
balsam queue submit \
  -n 1 -t 10 -q debug-flat-quad -A datascience \
  --site thetaknl_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# cooley
balsam queue submit \
  -n 1 -t 10 -q debug -A datascience \
  --site cooley_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# tom laptop
balsam queue submit \
  -n 1 -t 10 -q local -A local \
  --site tom_laptop \
  --tag workflow=hello_multi \
  --job-mode mpi \

# List queues
balsam queue ls
```

For a more detailed application, see the [hyperparameter optimization example](https://github.com/argonne-lcf/balsam_tutorial/blob/main/hyperopt/client_files/optimizer_test.py) in the Argonne github repository.

