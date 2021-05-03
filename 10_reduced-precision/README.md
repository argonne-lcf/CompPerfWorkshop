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
integer precision (`int8`), and custom formats designed for machine learning: "(Google)
Brain floating point format" (`bfloat16`) and TensorFloat-32 (`tf32`).

Deep learning models are often highly tolerant of reduced floating point precision during
training, and their model weights can typically be quantized to integers during inference
(after training).

Mixed precision, as the name implies, mixes variables of reduced and possibly single
precision in a single operation.

This presentation will discuss the use of reduced and mixed precison in TensorFlow and
PyTorch and explore the profiling of the models on the A100s.

![Comparison of significands](images/fp32_tf32_fp16_bfloat16.png)
