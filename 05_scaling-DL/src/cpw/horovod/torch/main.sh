#!/bin/bash

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

if [ $SHELL = "/bin/zsh" ]; then
  eval "$(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/bin/conda shell.zsh hook)"
else
  eval "$(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/bin/conda shell.bash hook)"
fi

python3 -m pip install --upgrade hydra-core hydra_colorlog

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)

LOGDIR="${DIR}/logs"
mkdir -p "${LOGDIR}"

LOGFILE="${LOGDIR}/${TSTAMP}.log"

NGPUS=$(nvidia-smi -L | wc -l)
NRANKS=$(cat $COBALT_NODEFILE | wc -l)

let "NPROCS = $NGPUS * $NRANKS"

echo "*************************************************************"
echo "STARTING A NEW RUN ON ${NPROCS} GPUs across ${NRANKS} RANKS"
echo "DATE: ${TSTAMP}"
echo "NGPUS: ${NGPUS}"
echo "NRANKS: ${NRANKS}"
echo "NPROCS = NGPUS * NRANKS = ${NPROC}"
echo "*************************************************************"

mpirun -np ${NPROCS} -hostfile ${COBALT_NODEFILE} -x PATH -x LD_LIBRARY_PATH python3 main.py
