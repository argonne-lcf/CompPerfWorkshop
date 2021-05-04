#!/bin/bash

## load TF module
module load conda/tensorflow/2021-03-02
conda activate

# set proxies
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128

## run the application
python ./tf2_mnist.py 
