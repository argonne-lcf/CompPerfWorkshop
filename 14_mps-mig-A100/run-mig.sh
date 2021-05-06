#!/bin/bash
  
CUDA_VISIBLE_DEVICES=<MIG-UUID> ./run_tf2_mnist.sh &
CUDA_VISIBLE_DEVICES=<MIG-UUID> ./run_pt_mnist.sh &

