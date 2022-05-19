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

