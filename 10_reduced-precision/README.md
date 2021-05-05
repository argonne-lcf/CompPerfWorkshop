# Reduced and Mixed Precision

The use of single precision (`float32`) instead of double precision (`float64`) floating
point quantities in simulation has long been used to accelerate scientific software (and
reduce energy consumption), when the corresponding loss in accuracy can be tolerated. The
design of modern CPUs and GPUs (and their memory subsystems) often means that a near 2x
speedup can be achieved simply by replacing all `double` variables with `float`.
Twice as many values can fit into the vector registers, and memory, memory bandwidth, and
cache pressure are all halved.
<!-- FPU add, multiply, ... should be roughly the same for single and double precision
in x86-64
https://stackoverflow.com/questions/4584637/double-or-float-which-is-faster
https://stackoverflow.com/questions/3426165/is-using-double-faster-than-float
-->

In the realm of deep learning, this strategy has enjoyed great success and has been
extended to more extreme forms of reduced precision, incluidng half precision (`float16`),
integer precision (`int8`, `int4`, `int1`), and custom types and modes designed for
machine learning: "(Google) Brain floating point format" (`bfloat16`) and TensorFloat-32
(`tf32`).

<!-- TensorFloat-32 is not really a floating point format or type, but rather a Tensor 
Core mode. See 2020 talk by Dusan Stosic -->

Deep learning models are often highly tolerant of reduced floating point precision during
training, and their model weights can typically be quantized to integers during inference
(after training).

Mixed precision, as the name implies, mixes variables of reduced and possibly single
precision in a single operation.

This presentation will discuss the use of reduced and mixed precison in TensorFlow and
PyTorch and explore the profiling of the models on the A100 GPUs on ThetaGPU.

![Comparison of significands](images/fp32_tf32_fp16_bfloat16.png)


## References

- [ECCV 2020 Tutorial: Accelerating Computer Vision with Mixed Precision](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/)

- [NERSC SC20 Tutorial: Deep Learning at
  Scale](https://github.com/NERSC/sc20-dl-tutorial#enabling-mixed-precision-training)
  
  
## Tensor Core utilization and changes in A100s

|| V100  | A100 |
|-------|:-------------:|:-------------:|
| CUDA Cores | 5120  | 6912  |
| Tensor Cores (TC) | 640  | 432  |
| TC modes | fp16 | fp16, bfloat16, tf32, fp64 | 

\* and `int8, int4, int1` for both generations. 

The 16:1 ratio of CUDA cores to Tensor cores in the Ampere SM and the 8:1 ratio in the
Volta SM are evident from their layouts:
![GA100 SM](./images/ga100-sm.png)
![GV100 SM](./images/gv100-sm.png)



## Automatic mixed precision


### Loss scaling

![Loss scaling](./images/loss-scaling.png)


<!-- ### Gradient clipping -->

## Tensor Float 32 (TF32) mode

![TensorFloat 32 input and output](./images/tf32-mode.png)

This is the default math mode for single precision training on A100s.

Disable it at the system level via `NVIDIA_TF32_OVERRIDE=0`, but:
> it must be set before the application is launched, as the effect of any change after the
application launch is unspecified. The variable affects only the mode of FP32
operations. Operations using FP64 or one of the 16-bit formats are not affected and
continue to use those corresponding types.




## Automatic mixed precision in deep learning frameworks: TensorFlow and PyTorch

Enablilng automatic `float16`/`float32` mixed precision in TensorFlow and PyTorch only
requires changes to a few lines of code, as shown in the previous section for TensorFlow
[Profiling: reduced precision](../09_profiling_frameworks/README.md).

Currently, `bfloat16` support is unavailable for NVIDIA GPUs in either TensorFlow and
PyTorch. However, TensorFlow supports automatic `bfloat16`/`float32` mixed precision for
Google TPUs. cuDNN has some convolutions with `bfloat16` now but it has not yet been
natively integrated into the production framework branches.

Continue on to the subfolders for more framework-specific details:
- [TensorFlow](./TensorFlow/README.md)
- [PyTorch](./PyTorch/README.md)

