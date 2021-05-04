#!/bin/bash

# load Torch module
module load conda/pytorch/2021-03-02 
conda activate

# set proxies
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128


APPDIR=/lus/theta-fs0/projects/datascience/memani/comp_perf_workshop/
cd $APPDIR

## run the application
python pt_mnist.py --epochs 10
