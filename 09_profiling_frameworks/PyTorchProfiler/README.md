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
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.01%     134.860us        56.49%        1.023s      51.149ms            20  
                aten::convolution         0.01%     141.280us        56.48%        1.023s      51.142ms            20  
               aten::_convolution         0.02%     384.628us        56.47%        1.023s      51.135ms            20  
         aten::mkldnn_convolution        56.41%        1.022s        56.45%        1.022s      51.116ms            20  
                 aten::batch_norm         0.01%     114.737us        25.80%     467.331ms      23.367ms            20  
     aten::_batch_norm_impl_index         0.01%     182.307us        25.80%     467.217ms      23.361ms            20  
          aten::native_batch_norm        25.59%     463.451ms        25.79%     467.005ms      23.350ms            20  
                      aten::relu_         0.02%     300.869us         8.25%     149.418ms       8.789ms            17  
                 aten::clamp_min_         0.01%     131.496us         8.23%     149.118ms       8.772ms            17  
                  aten::clamp_min         8.23%     148.986ms         8.23%     148.986ms       8.764ms            17  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.811s
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
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Source Location                                                              
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
              LABEL2: masking        84.66%        1.790s        99.94%        2.113s        2.113s             1  ...3/lib/python3.8/site-packages/torch/autograd/profiler.py(616): __enter__  
                                                                                                                                                                                                
                  aten::copy_         7.86%     166.179ms         7.86%     166.179ms     166.179ms             1  v0.py(30): forward                                                           
                                                                                                                                                                                                
                  aten::copy_         7.40%     156.435ms         7.40%     156.435ms     156.435ms             1  v0.py(31): forward                                                           
                                                                                                                                                                                                
                  aten::addmm         0.01%     276.443us         0.02%     463.267us     115.817us             4  ...mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1755): linear  
                                                                                                                                                                                                
              aten::clamp_min         0.01%     133.522us         0.02%     349.911us      43.739us             8  ...2/mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1206): relu  
                                                                                                                                                                                                
          LABEL1: linear pass         0.01%     119.615us         0.05%       1.113ms       1.113ms             1  ...3/lib/python3.8/site-packages/torch/autograd/profiler.py(616): __enter__  
                                                                                                                                                                                                
    aten::_local_scalar_dense         0.01%     116.550us         0.01%     116.550us     116.550us             1  v0.py(29): forward                                                           
                                                                                                                                                                                                
             aten::as_strided         0.00%      85.131us         0.00%      85.131us      10.641us             8  ...mconda3/lib/python3.8/site-packages/torch/nn/functional.py(1755): linear  
                                                                                                                                                                                                
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
Self CPU time total: 2.114s
```
The profile shows that the execution time of section `LABEL2: masking` takes 99.95% of total CPU time while in section `LABEL1: linear pass` code spends only 0.04%. Operations of copying tensors to the device at `v0.py(30): forward` and copying back at `v0.py(31): forward` take about 20% of execution time. We can optimize it if instead of `np.argwhere` do indexing on GPU with `torch.nonzero` - [example2/v1.py](example2/v1.py):
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
After this optimization total execution time was improved more than 100 times which is much better than just elimination of copy operation. The reason for that is that we computed `np.argwhere` on CPU while now we do this operation on GPU. PyTorch profile does not analyze NumPy operations so we missed them in the profile. 

### Python cProfile
PyTorch profile analyses only PyTorch operations which makes understanding of hotspots confusing. To profile all operations, one may use python profiler - [example2/v2.py](example2/v2.py):
```bash
         101 function calls (89 primitive calls) in 2.032 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     10/1    0.000    0.000    2.032    2.032 module.py:1009(_call_impl)
        1    0.084    0.084    2.032    2.032 v2.py:24(forward)
        1    0.000    0.000    1.658    1.658 <__array_function__ internals>:2(argwhere)
      4/1    0.000    0.000    1.658    1.658 {built-in method numpy.core._multiarray_umath.implement_array_function}
        1    0.002    0.002    1.658    1.658 numeric.py:537(argwhere)
        2    0.000    0.000    1.656    0.828 fromnumeric.py:52(_wrapfunc)
        1    0.000    0.000    1.298    1.298 <__array_function__ internals>:2(nonzero)
        1    0.000    0.000    1.298    1.298 fromnumeric.py:1816(nonzero)
        1    1.298    1.298    1.298    1.298 {method 'nonzero' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.358    0.358 <__array_function__ internals>:2(transpose)
        1    0.000    0.000    0.358    0.358 fromnumeric.py:601(transpose)
        1    0.000    0.000    0.358    0.358 fromnumeric.py:39(_wrapit)
        1    0.000    0.000    0.358    0.358 _asarray.py:14(asarray)
        1    0.357    0.357    0.357    0.357 {built-in method numpy.array}
        1    0.166    0.166    0.166    0.166 {method 'cpu' of 'torch._C._TensorBase' objects}
        1    0.123    0.123    0.123    0.123 {method 'cuda' of 'torch._C._TensorBase' objects}
        1    0.001    0.001    0.001    0.001 {method 'item' of 'torch._C._TensorBase' objects}
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

To see traces one has to install [PyTorch profiler TensorBoard Plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md). To do it on ThetaGPU you need to copy conda environment first (on ThetaGPU login node `thetagpusn1`):
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
ssh -L PORT:localhost:PORT username@theta.alcf.anl.gov ssh -L PORT:localhost:PORT thetagpusn1
```

Now you can open tensorboard in your browser `http://localhost:PORT`.
![tensorboard_overview](figs/profile.png)

More information on the example and usage of the new PyTorch profile can be found on its [github page](https://github.com/pytorch/kineto/tree/master/tb_plugin).
