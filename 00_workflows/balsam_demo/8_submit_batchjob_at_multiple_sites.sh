#!/bin/bash

# Submit BatchJobs at multiple sites
# thetagpu
# note: should use full-node queue
balsam queue submit \
  -n 1 -t 10 -q single-gpu -A training-gpu \
  --site thetagpu_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# theta knl
balsam queue submit \
  -n 1 -t 10 -q debug-flat-quad -A training-gpu \
  --site thetaknl_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# cooley
balsam queue submit \
  -n 1 -t 10 -q debug -A training-gpu \
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
