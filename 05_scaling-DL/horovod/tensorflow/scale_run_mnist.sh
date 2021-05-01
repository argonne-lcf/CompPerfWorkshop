#!/bin/bash

BATCH_SIZE=256
# EPOCHS=20
# LR=0.001
# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 ./torch_ddp_cifar10.py --gpu --lr=0.001 --batch_size=256 --epochs=10 > torch_ddp_gpu4.log&
# CUDA_VISIBLE_DEVICES0=0
# CUDA_VISIBLE_DEVICES1=0,1
# CUDA_VISIBLE_DEVICES2=0,1,2
# CUDA_VISIBLE_DEVICES3=0,1,2,3
# CUDA_VISIBLE_DEVICES4=0,1,2,3,4
# CUDA_VISIBLE_DEVICES5=0,1,2,3,4,5
# CUDA_VISIBLE_DEVICES6=0,1,2,3,4,5,6
# CUDA_VISIBLE_DEVICES7=0,1,2,3,4,5,6,7
# CUDA_VISIBLE_DEVICES8=0,1,2,3,4,5,6,7,8

for n in 1 2 4 8
do
    outfile=./scale_logs_mnist/tensorflow_hvd_mnist_batch${BATCH_SIZE}_gpu${n}.log
    echo ${outfile}
    mpirun -np ${n} --verbose python3 ./tf2_mnist.py --batch_size=$BATCH_SIZE --epochs=20 >& ${outfile}
    # python3 -m torch.distributed.launch --nproc_per_node=$n --nnodes=1 --node_rank=0 \
    #     ./torch_ddp_mnist.py --gpu --lr=0.001 --batch_size=$BATCH_SIZE --epochs=20 >& ${outfile}
done

