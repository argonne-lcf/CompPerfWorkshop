#!/bin/bash

export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    ./DDP/torch_ddp_mnist.py --gpu --epochs=0 --nosave &
