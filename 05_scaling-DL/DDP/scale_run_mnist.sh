#!/bin/bash

BATCH_SIZE=256
EPOCHS=50

for n in 1 2 4 8
do
    outfile=./results_mnist/torch_ddp_mnist_batch${BATCH_SIZE}_gpu${n}_epochs${EPOCHS}.log
    echo ${outfile}
    python3 -m torch.distributed.launch --nproc_per_node=$n --nnodes=1 --node_rank=0 \
        ./torch_ddp_mnist.py --gpu --lr=0.001 --batch_size=$BATCH_SIZE --epochs=$EPOCHS >& ${outfile}
done

