# PyTorch Profiler

Profiling is an efficient way of measuring the performance and doing optimization of your PyTorch scripts. It allows you to examine your script and understand if it has severe performance issues. One can measure and analyze execution time and memory consumption as well as finding bottlenecks and trace source code. In this tutorial we will discuss several ways to profile PyTorch scripts such as [`torch.autograd.profiler`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile), [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html#torch-profiler), [`torch.utils.bottleneck`](https://pytorch.org/docs/stable/bottleneck.html#torch-utils-bottleneck) and [Python cProfiler](https://docs.python.org/3/library/profile.html#module-cProfile).

In this tutorial we will:
* demonstrate PyTorch autograd profiler interface, measure CPU execution time and memory allocation
* profile simple performance issue on CPU
* profile simple performance issue on GPU
* compare PyTorch profiler with Python cProfiler and PyTorch bottleneck
* demonstrate new PyTorch profiler (introduced in PyTorch 1.8)

Table of Contents
* [Dependencies](#dependencies)
* [Demonstraion of the PyTorch profiler](#demonstraion-of-the-pytorch-profiler)
  - [Analysis of execution time](#analysis-of-execution-time)
  - [Analysis of memory allocation](#analysis-of-memory-allocation)
  - [Fixing performance issue](#fixing-performance-issue)
* [Example of profiling](#example-of-profiling)
  - [Python cProfile](#python-cprofile)
  - [PyTorch bottleneck](#pyTorch-bottleneck)
  - [Warnings](#warnings)
  - [New PyTorch profiler](#new-rytorch-profiler)


## Dependencies
Profiler is part of the PyTorch and can be used out of the box. PyTorch 1.8 [introduces new improved performance tool `torch.profiler`](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/). New profiler has a convenient dashboard in `tensorboard` so it has to be installed. Some examples in the tutorial use a model from `torchvision`.

These examples can be run on Theta GPUs. To do it you need to ssh to Theta:
```bash
ssh username@theta.alcf.anl.go
```
and ssh to ThetaGPU login node:
```bash
ssh thetagpusn1
```
where you can get a node in interactive regime:
```bash
qsub -I -n 1 -t 30 -A YoueProject -q single-gpu
```
Finally, you can activate conda environment
```bash
module load conda/pytorch
conda activate
```
and run all examples.


## Demonstraion of the PyTorch profiler
### Analysis of execution time
Let's start with analyzing execution time. We will use `resnet18` model from `torchvision` for demonstration.
```python
import torch
import torchvision.models as models
import torch.autograd.profiler as profiler

torch.set_default_tensor_type(torch.DoubleTensor)
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
```
One can profile execution with profiler as a context manager and print results:
```python
with profiler.profile() as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```
For convenience, this example is stored in [example/v0.py](example1/v0.py). Profiler output is presented below
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.12%     297.650us        75.64%     183.870ms       9.194ms            20  
                aten::convolution         0.13%     309.871us        75.52%     183.573ms       9.179ms            20  
               aten::_convolution         0.21%     512.073us        75.39%     183.263ms       9.163ms            20  
       aten::_convolution_nogroup         0.15%     359.211us        75.18%     182.751ms       9.138ms            20  
                aten::thnn_conv2d         0.10%     243.129us        75.01%     182.336ms       9.117ms            20  
        aten::thnn_conv2d_forward        21.84%      53.091ms        74.91%     182.093ms       9.105ms            20  
                     aten::addmm_        52.10%     126.652ms        52.10%     126.652ms       6.333ms            20  
                 aten::batch_norm         0.09%     212.893us        17.60%      42.787ms       2.139ms            20  
     aten::_batch_norm_impl_index         0.19%     469.533us        17.51%      42.574ms       2.129ms            20  
          aten::native_batch_norm        15.22%      37.000ms        17.30%      42.063ms       2.103ms            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 243.086ms
```
In the output, the function calls are sorted by total CPU time. It is important to note that `CPU total time` includes the time from all subroutines calls, but `Self CPU time` excludes it. For example, the total execution time of `aten::conv2d` consists of several operations `297.650us` and calling other functions which make in total 83.870ms. In opposite, in function `aten::addmm_` no time spend on calling subroutines. It is possible to sort results by another metric such as `self_cpu_time_total` or [other](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile.table).

Most time of execution was spent in convolutional layers. This model has several convolutions and one can examine different layers if sorted results by input tensor shape (another approach would be use labes, we will demonstrate it later) - [example/v1.py](example1/v1.py).
```python
with profiler.profile(record_shapes=True) as prof:
    model(inputs)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
```
In the output convolutions are grouped by input tensor shape (some columns deleted for a better presentation)
```bash
---------------------------------  ------------  -----------     ------------------------------
                             Name    Self CPU %  CPU total %     Input S
                     aten::conv2d         0.04%       21.61%     [[5,64, 64, 3, 3], [], [], [],  
                aten::convolution         0.02%       21.57%     [[5, 64, 4, 3, 3], [], [], [],  
               aten::_convolution         0.05%       21.55%     [[5, 64, 5, 3, 3], [], [], [],  
       aten::_convolution_nogroup         0.03%       21.50%     [[5, 64, 56, 56],  [], [], [],  
                aten::thnn_conv2d         0.02%       21.46%     [[5, 64, 4, 3, 3], [], [], [],  
        aten::thnn_conv2d_forward         7.80%       21.45%     [[5, 64, 56, 56],, [], [], [],  
                     aten::addmm_        13.48%       13.48%     [[6476], [576, 3136], [], [] 
                     aten::conv2d         0.01%       12.57%     [[5,12, 512, 3, 3], [], [], []  
                aten::convolution         0.01%       12.55%     [[5, 512,12, 3, 3], [], [], []  
               aten::_convolution         0.02%       12.54%     [[5, 512, 2, 3, 3], [], [], []  
---------------------------------  ------------  ------------    ------------------------------
Self CPU time total: 246.923ms
```
### Analysis of memory allocation
Profiler also allows to analyze the memory allocated in different parts of the model. Similar to CPU execution time, 'self' memory accounts for memory allocated in the function excluding calls of subroutines. The profiler will analyze memory if attibute `profile_memory=True` is set  - [example1/v2.py](example1/v2.py).
```python
with profiler.profile(profile_memory=True) as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```
As a result of the profile, one could see
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    aten::resize_         0.43%       1.019ms         0.43%       1.019ms      24.257us     670.40 Mb     670.40 Mb            42  
                      aten::empty         0.90%       2.149ms         0.90%       2.149ms      14.926us      94.84 Mb      94.84 Mb           144  
                      aten::addmm         0.12%     286.240us         0.13%     313.193us     313.193us      39.06 Kb      39.06 Kb             1  
                        aten::add         0.26%     626.444us         0.26%     626.444us      31.322us         160 b         160 b            20  
              aten::empty_strided         0.00%       3.624us         0.00%       3.624us       3.624us           8 b           8 b             1  
                     aten::conv2d         0.07%     161.425us        76.35%     182.082ms       9.104ms     655.09 Mb           0 b            20  
                aten::convolution         0.09%     205.545us        76.28%     181.921ms       9.096ms     655.09 Mb           0 b            20  
               aten::_convolution         0.20%     469.240us        76.20%     181.715ms       9.086ms     655.09 Mb           0 b            20  
       aten::_convolution_nogroup         0.12%     280.513us        76.00%     181.246ms       9.062ms     655.09 Mb           0 b            20  
          aten::_nnpack_available         0.02%      49.942us         0.02%      49.942us       2.497us           0 b           0 b            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 238.486ms
```

### Fixing performance issue
A careful reader could notice that PyTorch used the native algorithm `aten::thnn_conv2d` for convolution layers. Although, for execution on CPU PyTorch is optimized with [MKLDNN library](https://github.com/rsdubtso/mkl-dnn) and should have used the corresponding convolution. This issue could reduce performance. In this example, PyTorch used a native algorithm because the convolution algorithm in double precision is missing in MKLDNN, so switching to float precision will turn MKLDNN on - [example1/v3.py](example1/v3.py):
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.14%     130.420us        64.71%      60.008ms       3.000ms            20  
                aten::convolution         0.12%     106.667us        64.57%      59.878ms       2.994ms            20  
               aten::_convolution         0.37%     345.810us        64.46%      59.771ms       2.989ms            20  
         aten::mkldnn_convolution        63.72%      59.083ms        64.09%      59.425ms       2.971ms            20  
                 aten::batch_norm         0.15%     139.862us        21.30%      19.752ms     987.623us            20  
     aten::_batch_norm_impl_index         0.20%     189.244us        21.15%      19.613ms     980.630us            20  
          aten::native_batch_norm        17.24%      15.987ms        20.90%      19.385ms     969.236us            20  
                 aten::max_pool2d         0.03%      30.184us         8.70%       8.069ms       8.069ms             1  
    aten::max_pool2d_with_indices         8.61%       7.988ms         8.67%       8.038ms       8.038ms             1  
                     aten::select         1.96%       1.815ms         3.28%       3.041ms       6.992us           435  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 92.728ms
```
As a result, `aten::mkldnn_convolution` was used. Due to optimizations in MKLDNN, execution time was decreased more than twice (92.728ms vs 243.086ms).

## Example of profiling
In this section let's build a simple model and profile it. We build a model which takes a tensor of size 512, does several linear transformations with activation, and calculates a threshold. We want to compare this threshold with our mask and get indexes:
```python
import torch
import numpy as np
import torch.autograd.profiler as profiler


class MyModule(torch.nn.Module):
    def __init__(
            self, in_features: int,
            out_features: int,
            hidden_sizes: list,
            bias: bool = True):
        super(MyModule, self).__init__()

        sizes = [in_features] + hidden_sizes + [out_features]
        layers = []
        for s in range(len(sizes)-1):
            layers.append(torch.nn.Linear(sizes[s], sizes[s+1], bias))
            layers.append(torch.nn.ReLU())
        self.linear = torch.nn.Sequential(*layers)

    def forward(self, input, mask):
        with profiler.record_function("LABEL1: linear pass"):
            out = self.linear(input)

        with profiler.record_function("LABEL2: masking"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx
```
At this time we demonstrate measurement of the time spent in different sections of the model. We marked linear and masking sections with labels. One can build an instance of this model and run profiler - [example2/v0.py](example2/v0.py):
```python
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = MyModule(512, 8, [32, 32, 32])
input = torch.rand(512 * 512, 512)
mask = torch.rand((512, 512, 512))

# warm-up
model(input, mask)

with profiler.profile(with_stack=True) as prof:
    out, idx = model(input, mask)
print(prof.key_averages(group_by_stack_n=1).table(sort_by='self_cpu_time_total', row_limit=5))
```
This time we execute modes on GPU. At the first call, CUDA does some benchmarking and chose the best algorithm for convolutions, therefore we need to warm up CUDA to ensure accurate performance benchmarking. Also, we used the flag `with_stack=True` which makes it possible to track the place in sources where the function was called. 
```bash
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Source Location                                                              
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
              LABEL2: masking        85.48%        2.440s        99.94%        2.853s        2.853s             1  ...8/lib/python3.8/site-packages/torch/autograd/profiler.py(616): __enter__  
                                                                                                                                                                                                
                  aten::copy_         7.83%     223.551ms         7.83%     223.551ms     223.551ms             1  example2/v0.py(28): forward                                                  
                                                                                                                                                                                                
                  aten::copy_         6.54%     186.715ms         6.54%     186.715ms     186.715ms             1  example2/v0.py(27): forward                                                  
                                                                                                                                                                                                
    aten::_local_scalar_dense         0.08%       2.177ms         0.08%       2.177ms       2.177ms             1  example2/v0.py(26): forward                                                  
                                                                                                                                                                                                
                  aten::addmm         0.01%     289.035us         0.02%     481.874us     120.468us             4  ...orch1.8/lib/python3.8/site-packages/torch/nn/functional.py(1753): linear  
                                                                                                                                                                                                
          LABEL1: linear pass         0.01%     193.168us         0.05%       1.340ms       1.340ms             1  ...8/lib/python3.8/site-packages/torch/autograd/profiler.py(616): __enter__  
                                                                                                                                                                                                
                   aten::relu         0.01%     153.368us         0.01%     317.394us      79.349us             4  .../torch1.8/lib/python3.8/site-packages/torch/nn/functional.py(1206): relu  
                                                                                                                                                                                                
                  aten::empty         0.01%     146.710us         0.01%     146.710us     146.710us             1  example2/v0.py(28): forward                                                  
                                                                                                                                                                                                
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
Self CPU time total: 2.855s
```
The profile shows that the execution time of section `LABEL2: masking` takes 99.95% of total CPU time while in section `LABEL1: linear pass` code spends only 0.04%. Operations of copying tensors to the device at `example2/v0.py(28): forward` and copying back at `example2/v0.py(29): forward` take about 20% of execution time. We can optimize it if instead of `np.argwhere` do indexing on GPU with `torch.nonzero` - [example2/v1.py](example2/v1.py):
```python
    def forward(self, input, mask):
        with profiler.record_function("LABEL1: linear pass"):
            out = self.linear(input)

        with profiler.record_function("LABEL2: masking"):
            threshold = out.sum(axis=1).mean()  # removed.item()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)
        return out, hi_idx
```
```bash
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          aten::nonzero        96.00%      15.948ms        96.08%      15.962ms      15.962ms             1  
    LABEL1: linear pass         0.82%     136.417us         2.67%     442.910us     442.910us             1  
            aten::addmm         0.78%     130.082us         0.93%     153.704us      38.426us             4  
        LABEL2: masking         0.34%      56.329us        96.98%      16.112ms      16.112ms             1  
        aten::threshold         0.26%      43.308us         0.33%      54.649us      13.662us             4  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 16.613ms
```
After this optimization total execution time was improved more than 100 times which is much better than just elimination of copy operation. The reason for that is that we computed `np.argwhere` on CPU while now we do this operation on GPU. PyTorch profile does not analyze NumPy operations so we missed them in the profile. 

### Python cProfile
PyTorch profile analyses only PyTorch operations which makes understanding of hotspots confusing. To profile all operations, one may use python profiler - [example2/v2.py](example2/v2.py):
```bash
         181 function calls (169 primitive calls) in 3.434 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     10/1    0.000    0.000    3.434    3.434 module.py:866(_call_impl)
        1    0.078    0.078    3.434    3.434 v2.py:21(forward)
        1    0.000    0.000    2.817    2.817 <__array_function__ internals>:2(argwhere)
      4/1    0.000    0.000    2.817    2.817 {built-in method numpy.core._multiarray_umath.implement_array_function}
        1    0.005    0.005    2.817    2.817 numeric.py:537(argwhere)
        2    0.000    0.000    2.812    1.406 fromnumeric.py:52(_wrapfunc)
        1    0.000    0.000    1.476    1.476 <__array_function__ internals>:2(nonzero)
        1    0.000    0.000    1.476    1.476 fromnumeric.py:1816(nonzero)
        1    1.476    1.476    1.476    1.476 {method 'nonzero' of 'numpy.ndarray' objects}
        1    0.000    0.000    1.335    1.335 <__array_function__ internals>:2(transpose)
        1    0.000    0.000    1.335    1.335 fromnumeric.py:601(transpose)
        1    0.000    0.000    1.335    1.335 fromnumeric.py:39(_wrapit)
        1    0.000    0.000    1.335    1.335 _asarray.py:14(asarray)
        1    1.335    1.335    1.335    1.335 {built-in method numpy.array}
        1    0.348    0.348    0.348    0.348 {method 'cuda' of 'torch._C._TensorBase' objects}
        1    0.188    0.188    0.188    0.188 {method 'cpu' of 'torch._C._TensorBase' objects}
        1    0.003    0.003    0.003    0.003 {method 'item' of 'torch._C._TensorBase' objects}

```
Most of the time `2.817s` was spent in `argwhere` function which we compute on CPU. Moreover, this function called `{method 'nonzero' of 'numpy.ndarray' objects}` so our optimization was natural.


### PyTorch bottleneck
Modifying your script for running python cProfiler or PyTorch profiler with different arguments could be discouraging and is not necessary for the first step. There is a convenient and simple tool called [`torch.utils.bottleneck`](https://pytorch.org/docs/stable/bottleneck.html#torch-utils-bottleneck) which can be used with your script with no modification. It summarizes the analysis of your script with both: Python cProfiler and PyTorch profiler. Using it for fine-tuning is not a good practice (because of missing warm-up, measuring initialization, etc.) but is a good initial step for debugging bottlenecks in your script.
```bash
python -m torch.utils.bottleneck example2/v3.py
```
The output of the bottleneck is too big to show it here. Basically, it combines outputs of python cProfile, `torch.autorgrad.profile` for CPU and CUDA modes.

### Warnings
* The launch of CUDA kernels is asynchronous, so if you want to measure time spent in them make sure that you turned flag `use_cuda` on. Otherwise, your results may be misleading.
* While profile collects events and analyses them it has a huge overhead. Profiler is helpful in searching for performance issues but slows down training/evaluation. Be sure that you removed it when you finish your code investigation.


## New PyTorch profiler
With PyTorch 1.8 release [the new PyTorch profiler was introduced](https://pytorch.org/docs/stable/profiler.html#torch-profiler). It is the next version of `torch.autorgrad.profile` and will replace it in future releases. The new `torch.profile` has a different [API](https://pytorch.org/docs/stable/profiler.html#torch-profiler) but it can be used instead of  `torch.autorgrad.profile`: collect and print profile. But, more interestingly, it provides a convenient dashboard with a summary of all events and recommendations for optimization - [example3/v0.py](example3/v0.py)
```python
import torch
import torchvision.models as models

torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

dir_name = './'

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)
        ) as prof:

    for _ in range(8):
        model(inputs)
        prof.step()
```
In this example, we collect activities on both CPU and GPU. Due to `schedule` argument, we can use `torch.profiler.schedule` which with `wait=0` skip no iterations, `warmup=1` start warming up on first, `active=3` record second - fourth iteration, and when the trace becomes available `torch.profiler.tensorboard_trace_handler` is called to save a trace. This cycle repeats with the fifth iteration so in our example two traces will be saved. After execution, we will have `some_name.pt.trace.json` and `some_name_2.pt.trace.json` traces saved.

To see traces one has to install [PyTorch profiler TensorBoard Plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md). To do it on ThetaGPU you need to copy conda environment first (on ThetaGPU login node):
```bash
module load conda/pytorch
conda activate
conda create --clone $CONDA_PREFIX --name yourEvnName
conda activate yourEvnName
pip install torch-tb-profiler
```
and run tensorboard with specifying `logdir` where your traces are stored. You can run tensorboard on `thetagpusn1` node:
```bash
tensorboard --port <PORT> --bind_all --logdir </path/to/log/output/>
```
Also, in this case, you will need to do some ssh port forwarding to access the server. On your local machine run
```bash
ssh -L localhost:PORT:localhost:PORT username@theta.alcf.anl.gov
```
Once you logged in to Theta login node you need to forward ports from the node where you run tensorboard to Theta login node (on Theta login node):
```bash
ssh -L localhost:PORT:localhost:PORT thetagpusn1
```

Now you can open tensorboard in your browser `http://localhost:PORT`.
![tensorboard_overview](figs/profile.png)

More information on the example and usage of the new PyTorch profile can be found on its [githab page](https://github.com/pytorch/kineto/tree/master/tb_plugin).
