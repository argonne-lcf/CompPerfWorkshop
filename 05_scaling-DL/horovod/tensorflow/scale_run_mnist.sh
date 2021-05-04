#!/bin/bash

BATCH_SIZE=256
EPOCHS=50

for n in 1 2 4 8
do
    outfile=./tf2_hvd_mnist_batch${BATCH_SIZE}_gpu${n}_epochs${EPOCHS}.log
    echo ${outfile}
    mpirun -np ${n} --verbose python3 ./tf2_hvd_mnist.py --batch_size=$BATCH_SIZE --epochs=$EPOCHS >& ${outfile}
done

