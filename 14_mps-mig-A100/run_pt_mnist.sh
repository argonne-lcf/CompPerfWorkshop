#!/bin/bash

# load Torch module
module load conda/pytorch/2021-03-02 
conda activate

# set proxies or use --attrs pubnet=true while requesting a node
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128

## run the application
python ./pt_mnist.py --epochs 10  > pt_mnist.log 2>&1
