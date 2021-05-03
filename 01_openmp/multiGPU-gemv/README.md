Last update 2021 May 3rd

Need to load llvm module on ThetaGPU
```
module load llvm/release-12.0.0
```

Case 1: offload one matrix-vector multiplication to a single GPU.

Case 2: offload multiple matrix-vector multiplication to a single GPU.
Multiple host threads launch target offload independently to keep GPU busy.

Case 3: offload multiple matrix-vector multiplication to multiple GPUs.
Multiple host threads launch target offload independently to on-node GPUs round-robin.
