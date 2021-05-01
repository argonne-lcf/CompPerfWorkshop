#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    ./DDP/torch_ddp_mnist.py --gpu --epochs=0 --nosave &
