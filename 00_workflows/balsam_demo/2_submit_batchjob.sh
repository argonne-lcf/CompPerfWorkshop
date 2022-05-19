#!/bin/bash
balsam queue submit \
  -n 1 -t 10 -q full-node -A datascience \
  --site thetagpu_tutorial \
  --tag workflow=hello \
  --job-mode mpi 
