#!/bin/bash

BATCH_SIZE=256
# EPOCHS=20
# LR=0.001
# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 ./torch_ddp_cifar10.py --gpu --lr=0.001 --batch_size=256 --epochs=10 > torch_ddp_gpu4.log&

for n in 1 2 4 8
do
    outfile=./results_mnist/torch_ddp_mnist_batch${BATCH_SIZE}_gpu${n}.log
    echo ${outfile}
    python3 -m torch.distributed.launch --nproc_per_node=$n --nnodes=1 --node_rank=0 \
        ./torch_ddp_mnist.py --gpu --lr=0.001 --batch_size=$BATCH_SIZE --epochs=200 >& ${outfile}
done

