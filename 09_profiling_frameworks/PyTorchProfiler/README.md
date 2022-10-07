# PyTorch Profiler

Profiling is an efficient way of measuring the performance and making optimization of your PyTorch scripts. It allows one to examine a PyTorch script and understand if it has some performance issues. One can measure and analyze execution time and memory consumption as well as finding bottlenecks and trace source code. In this tutorial we discuss several ways to profile PyTorch scripts such as [`torch.autograd.profiler`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile), [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html#torch-profiler), [`torch.utils.bottleneck`](https://pytorch.org/docs/stable/bottleneck.html#torch-utils-bottleneck) and [Python cProfiler](https://docs.python.org/3/library/profile.html#module-cProfile).

In this tutorial we:
* demonstrate PyTorch autograd profiler interface, measure CPU execution time and memory allocation
* profile simple performance issue on CPU
* profile simple performance issue on GPU
* compare PyTorch profiler with Python cProfiler and PyTorch bottleneck
* demonstrate visual PyTorch profiler with tensorboard plugin (introduced in PyTorch 1.8)

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
* [New PyTorch profiler](#new-pytorch-profiler)


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
module load conda/2021-11-30
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
inputs = torch.randn(128, 3, 224, 224)
```
One can profile execution with profiler as a context manager and print results:
```python
with profiler.profile() as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```
For convenience, this example is stored in [example1/v0.py](example1/v0.py). Profiler output is presented below
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.00%     205.227us        80.65%        4.745s     237.245ms            20  
                aten::convolution         0.00%     151.037us        80.65%        4.745s     237.234ms            20  
               aten::_convolution         0.00%     286.113us        80.65%        4.745s     237.227ms            20  
       aten::_convolution_nogroup         0.00%     227.487us        80.64%        4.744s     237.212ms            20  
                aten::thnn_conv2d         0.00%     148.750us        80.64%        4.744s     237.198ms            20  
        aten::thnn_conv2d_forward        30.72%        1.807s        80.64%        4.744s     237.191ms            20  
                     aten::addmm_        49.69%        2.923s        49.69%        2.923s      36.539ms            80  
                 aten::batch_norm         0.00%     154.610us        13.06%     768.367ms      38.418ms            20  
     aten::_batch_norm_impl_index         0.00%     231.829us        13.06%     768.212ms      38.411ms            20  
          aten::native_batch_norm        12.97%     762.916ms        13.05%     767.934ms      38.397ms            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 5.883s
```
In the output, the function calls are sorted by total CPU time. It is important to note that `CPU total time` includes the time from all subroutines calls, but `Self CPU time` excludes it. For example, the total execution time of `aten::conv2d` consists of several operations `297.650us` and calling other functions which make in total 183.870ms. In opposite, in function `aten::addmm_` no time spend on calling subroutines. It is possible to sort results by another metric such as `self_cpu_time_total` or [other](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile.table).

Most time of execution was spent in convolutional layers. This model has several convolutions and one can examine different layers if sorted results by input tensor shape (another approach would be use labes, we will demonstrate it later) - [example1/v1.py](example1/v1.py).
```python
with profiler.profile(record_shapes=True) as prof:
    model(inputs)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
```
In the output convolutions are grouped by input tensor shape
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls                                   Input Shapes  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
                     aten::conv2d         0.00%      50.848us        29.83%        1.707s     426.702ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
                aten::convolution         0.00%      71.394us        29.83%        1.707s     426.689ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
               aten::_convolution         0.00%      95.369us        29.83%        1.707s     426.671ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
       aten::_convolution_nogroup         0.00%      58.251us        29.83%        1.707s     426.647ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
                aten::thnn_conv2d         0.00%      40.978us        29.83%        1.707s     426.631ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
        aten::thnn_conv2d_forward        10.68%     611.171ms        29.83%        1.706s     426.621ms             4  [[128, 64, 56, 56], [64, 64, 3, 3], [], [], [  
                     aten::addmm_        19.12%        1.094s        19.12%        1.094s      68.352ms            16   [[64, 3136], [64, 576], [576, 3136], [], []]  
                     aten::conv2d         0.00%      29.956us        15.41%     881.441ms     293.814ms             3  [[128, 128, 28, 28], [128, 128, 3, 3], [], []  
                aten::convolution         0.00%      26.290us        15.41%     881.411ms     293.804ms             3  [[128, 128, 28, 28], [128, 128, 3, 3], [], []  
               aten::_convolution         0.00%      60.715us        15.41%     881.385ms     293.795ms             3  [[128, 128, 28, 28], [128, 128, 3, 3], [], []  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
Self CPU time total: 5.721s

```
### Analysis of memory allocation
Profiler also allows to analyze the memory allocated in different parts of the model. Similar to CPU execution time, 'self' memory accounts for memory allocated in the function excluding calls of subroutines. The profiler will analyze memory if attibute `profile_memory=True` is set  - [example1/v2.py](example1/v2.py).
```python
with profiler.profile(profile_memory=True) as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```
You will see some warning of the profiler mentioning that not all memory allocation/deallocation events are analyzed. This happens because we profile only model forward pass and some allocations in the initialization are missed. You will see the following profile:
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    aten::resize_         0.01%     915.645us         0.01%     915.645us      21.801us      16.76 Gb      16.76 Gb            42  
                      aten::empty         0.11%       7.330ms         0.11%       7.330ms      50.904us       2.37 Gb       2.37 Gb           144  
                      aten::addmm         0.02%       1.443ms         0.02%       1.551ms       1.551ms    1000.00 Kb    1000.00 Kb             1  
                        aten::add         0.02%       1.407ms         0.02%       1.407ms      70.373us         160 b         160 b            20  
              aten::empty_strided         0.00%       6.052us         0.00%       6.052us       6.052us           8 b           8 b             1  
                     aten::conv2d         0.00%     251.835us        71.89%        4.664s     233.224ms      16.38 Gb           0 b            20  
                aten::convolution         0.00%     216.004us        71.88%        4.664s     233.211ms      16.38 Gb           0 b            20  
               aten::_convolution         0.01%     345.879us        71.88%        4.664s     233.200ms      16.38 Gb           0 b            20  
       aten::_convolution_nogroup         0.00%     249.293us        71.87%        4.664s     233.183ms      16.38 Gb           0 b            20  
          aten::_nnpack_available         0.00%      46.784us         0.00%      46.784us       2.339us           0 b           0 b            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 6.489s
```

### Fixing performance issue
A careful reader could notice that PyTorch used the native algorithm `aten::thnn_conv2d` for convolution layers. Although, for execution on CPU PyTorch is optimized with [MKLDNN library](https://github.com/rsdubtso/mkl-dnn) and should have used the corresponding convolution. This issue could reduce performance. In this example, PyTorch used a native algorithm because the convolution algorithm in double precision is missing in MKLDNN, so switching to float precision will turn MKLDNN on - [example1/v3.py](example1/v3.py):
```bash
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.01%     125.590us        64.83%        1.178s      58.914ms            20  
                aten::convolution         0.01%     134.744us        64.82%        1.178s      58.908ms            20  
               aten::_convolution         0.02%     346.988us        64.82%        1.178s      58.901ms            20  
         aten::mkldnn_convolution        64.76%        1.177s        64.80%        1.178s      58.884ms            20  
                 aten::batch_norm         0.01%     124.978us        17.24%     313.257ms      15.663ms            20  
     aten::_batch_norm_impl_index         0.01%     185.441us        17.23%     313.132ms      15.657ms            20  
          aten::native_batch_norm        17.02%     309.403ms        17.22%     312.915ms      15.646ms            20  
                      aten::relu_         0.02%     301.198us        11.30%     205.418ms      12.083ms            17  
                 aten::clamp_min_         0.01%     136.508us        11.29%     205.117ms      12.066ms            17  
                  aten::clamp_min        11.28%     204.981ms        11.28%     204.981ms      12.058ms            17  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.817s
```
As a result, `aten::mkldnn_convolution` was used. Due to optimizations in MKLDNN, execution time was decreased more than twice (1.817s vs 5.883s).

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
This time we execute models on GPU. At the first call, CUDA does some benchmarking and chose the best algorithm for convolutions, therefore we need to warm up CUDA to ensure accurate performance benchmarking. Also, we used the flag `with_stack=True` which makes it possible to track the place in sources where the function was called. 
```bash
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Source Location                                                              
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                                        LABEL2: masking        83.65%        1.980s        99.93%        2.365s        2.365s             1  ...3/lib/python3.8/site-packages/torch/autograd/profiler.py(435): __enter__  
                                                                                                                                                                                                                          
                                        cudaMemcpyAsync        16.26%     384.874ms        16.26%     384.874ms     128.291ms             3                                                                               
                                                                                                                                                                                                                          
                                            aten::empty         0.02%     522.000us         0.02%     522.000us     261.000us             2  ...a3/lib/python3.8/site-packages/torch/autograd/profiler.py(432): __init__  
                                                                                                                                                                                                                          
                                            aten::addmm         0.01%     238.000us         0.02%     364.000us      91.000us             4  ...mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1848): linear  
                                                                                                                                                                                                                          
                                    LABEL1: linear pass         0.01%     218.000us         0.04%       1.061ms       1.061ms             1  ...3/lib/python3.8/site-packages/torch/autograd/profiler.py(435): __enter__  
                                                                                                                                                                                                                          
                                        aten::clamp_min         0.01%     154.000us         0.01%     326.000us      40.750us             8  ...0/mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1299): relu  
                                                                                                                                                                                                                          
                                       cudaLaunchKernel         0.01%     145.000us         0.01%     145.000us       9.062us            16                                                                               
                                                                                                                                                                                                                          
                                           aten::linear         0.00%      77.000us         0.02%     562.000us     140.500us             4  ...mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1848): linear  
                                                                                                                                                                                                                          
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
Self CPU time total: 2.367s

```
The profile shows that the execution time of section `LABEL2: masking` takes 99.93% of total CPU time while in section `LABEL1: linear pass` code spends only 0.04%. Operations of copying tensors `cudaMemcpyAsync` take about 16% of execution time. We can optimize it if instead of `np.argwhere` do indexing on GPU with `torch.nonzero` - [example2/v1.py](example2/v1.py):
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
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        cudaMemcpyAsync        45.20%       1.176ms        45.20%       1.176ms       1.176ms             1  
                                            aten::empty        18.33%     477.000us        18.33%     477.000us      59.625us             8  
                                            aten::addmm         7.34%     191.000us        10.11%     263.000us      65.750us             4  
                                       cudaLaunchKernel         5.88%     153.000us         5.88%     153.000us       6.955us            22  
                                    LABEL1: linear pass         5.03%     131.000us        22.41%     583.000us     583.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.602ms
```
After this optimization total execution time was improved by O(100) times which is much better than just elimination of copy operation. The reason for that is that we computed `np.argwhere` on CPU while now we do this operation on GPU. PyTorch profile does not analyze NumPy operations so we missed them in the profile. 

### Python cProfile
PyTorch profile analyses only PyTorch operations which makes understanding of hotspots confusing. To profile all operations, one may use python profiler - [example2/v2.py](example2/v2.py):
```bash
         101 function calls (89 primitive calls) in 2.422 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     10/1    0.000    0.000    2.422    2.422 module.py:1096(_call_impl)
        1    0.077    0.077    2.422    2.422 v2.py:24(forward)
        1    0.000    0.000    1.937    1.937 <__array_function__ internals>:2(argwhere)
      4/1    0.000    0.000    1.937    1.937 {built-in method numpy.core._multiarray_umath.implement_array_function}
        1    0.004    0.004    1.937    1.937 numeric.py:570(argwhere)
        2    0.000    0.000    1.933    0.967 fromnumeric.py:52(_wrapfunc)
        1    0.000    0.000    1.325    1.325 <__array_function__ internals>:2(nonzero)
        1    0.000    0.000    1.325    1.325 fromnumeric.py:1827(nonzero)
        1    1.325    1.325    1.325    1.325 {method 'nonzero' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.608    0.608 <__array_function__ internals>:2(transpose)
        1    0.000    0.000    0.608    0.608 fromnumeric.py:602(transpose)
        1    0.000    0.000    0.608    0.608 fromnumeric.py:39(_wrapit)
        1    0.000    0.000    0.608    0.608 _asarray.py:23(asarray)
        1    0.608    0.608    0.608    0.608 {built-in method numpy.array}
        1    0.211    0.211    0.211    0.211 {method 'cuda' of 'torch._C._TensorBase' objects}
        1    0.195    0.195    0.195    0.195 {method 'cpu' of 'torch._C._TensorBase' objects}
        1    0.001    0.001    0.001    0.001 {method 'item' of 'torch._C._TensorBase' objects}
```
Most of the time `1.937s` was spent in `argwhere` function which we compute on CPU. Also, from this profile it is clear that `argwhere` called `{method 'nonzero' of 'numpy.ndarray' objects}` so our optimization was natural.


### PyTorch bottleneck
Modifying your script for running python cProfiler or PyTorch profiler with different arguments could be discouraging and is not necessary for the first step. There is a convenient and simple tool called [`torch.utils.bottleneck`](https://pytorch.org/docs/stable/bottleneck.html#torch-utils-bottleneck) which can be used with your script with no modification. It summarizes the analysis of your script with both: Python cProfiler and PyTorch profiler. Using it for fine-tuning is not a good practice (because of missing warm-up, measuring initialization, etc.) but is a good initial step for debugging bottlenecks in your script.
```bash
python -m torch.utils.bottleneck example2/v3.py
```
The output of the bottleneck is too big to show it here. Basically, it combines outputs of python cProfile, `torch.autorgrad.profile` for CPU and CUDA modes.

### Warnings
* The launch of CUDA kernels is asynchronous, so if you want to measure time spent in them make sure that you turned flag `use_cuda` on. Otherwise, your results may be misleading.
* While profile collects events and analyses them it has a huge overhead. Profiler is helpful in searching for performance issues but slows down training/evaluation. Be sure that you removed it when you finish your code investigation.


## (new) PyTorch profiler
With PyTorch 1.8 release [the new PyTorch profiler was introduced](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/). It is the next version of `torch.autorgrad.profile` and will replace it in future releases. The new `torch.profile` has a different [API](https://pytorch.org/docs/stable/profiler.html#torch-profiler) but it can be used instead of  `torch.autorgrad.profile`: collect and print profile. But, more interestingly, it provides a convenient dashboard with a summary of all events and recommendations for optimization - [example3/v0.py](example3/v0.py)
```python
import torch
import torchvision.models as models

torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = models.resnet18()
inputs = torch.randn(32, 3, 224, 224)

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
In this example, we collect activities on both CPU and GPU. Due to `schedule` argument, we can use `torch.profiler.schedule` which with `wait=0` skips no iterations, `warmup=1` starts warming up on first, `active=3` records second - fourth iterations, and when the trace becomes available `torch.profiler.tensorboard_trace_handler` is called to save a trace. This cycle repeats with the fifth iteration so in our example two traces will be saved. After execution, we will have `some_name.pt.trace.json` and `some_name_2.pt.trace.json` traces saved.

To see traces one has to install [PyTorch profiler TensorBoard Plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md). This can be done either by installing this package locally
```bash
module load conda/pytorch
conda activate
pip install --user torch-tb-profiler
```
or by copying conda environment
```bash
module load conda/pytorch
conda activate
conda create --clone $CONDA_PREFIX --name yourEvnName
conda activate yourEvnName
pip install torch-tb-profiler
```

This plugin allows exemining PyTorch profiler result in TensorBoard. To do so one have to run tensorboard with specifying `logdir` where your traces are stored. If you run tensorboard on `thetagpusn1` node:
```bash
tensorboard --port <PORT> --bind_all --logdir </path/to/log/output/>
```
you will need to do ssh port forwarding to access the server. On your local machine run
```bash
ssh -L PORT:localhost:PORT username@theta.alcf.anl.gov ssh -L PORT:localhost:PORT thetagpusn1
```

Now you can open tensorboard in your browser `http://localhost:PORT`.
![tensorboard_overview](figs/profile.png)

More information on the example and usage of the new PyTorch profile can be found on its [github page](https://github.com/pytorch/kineto/tree/master/tb_plugin).


## References and additional information
The following materials were used in this tutorial and recommended for further study.
- [PyTorch autograd profiler example](https://pytorch.org/tutorials/beginner/profiler.html)
- [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch profiler with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?highlight=tensorboard)
- [PyTorch TensorBoard Plugin usage instructions](https://github.com/pytorch/kineto/tree/main/tb_plugin)
