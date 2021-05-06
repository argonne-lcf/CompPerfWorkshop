# Mixed Precision in TensorFlow

Here we will build on the GAN example from the previous profiling module.  We will by
default run the training for 4 epochs, each consisting of 14 iterations when the batch
size is 4096.

We will use the April 2021 release NVIDIA-optimiized NGC Singularity container for
TensorFlow 2.x in this walkthrough:
```
singularity exec --nv -B /lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_21.04-py3.simg bash
```

<!-- ```
singularity exec --nv -B /lus /grand/projects/Comp_Perf_Workshop/containers/tf2_cpw.simg bash
``` -->

## Custom loss scaling

When discussing [reduced precision](../../09_profiling_frameworks/TensorFlow/reduced_precision/README.md) in the
profiling module, it was claimed that only one additional line was necessary to enable
mixed precision in Keras. That was not quite true; since we only cared about raw
performance, we did not even consider the effects on model accuracy. In general, loss
scaling is a necessary addition to your deep learning code in order to ensure
that mixed precision does not harm training. 
In [`train_GAN_optimized.py`](train_GAN_optimized.py), 
we have added a switch `use_scaled_loss` at the top of the file to enable loss scaling, which does the
following:
1. Scale up the loss by the adaptive factor during the forward pass:
```python
        #Update the generator:
        with tf.GradientTape() as tape:
                loss = forward_pass(
                    models["generator"],
                    models["discriminator"],
                    _input_size = 100,
                    _real_batch = data,
                )
				# new line; not used if use_scaled_loss=False
                scaled_gen_loss = _opts["generator"].get_scaled_loss(loss['generator'])
```
2. Undo the scaling to the grads and apply to the unscaled gradient updates to the model weights
```python
        # Apply the update to the network (one at a time):
        if use_scaled_loss:
            scaled_grads = tape.gradient(scaled_gen_loss, trainable_vars)
            grads = _opts["generator"].get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss["generator"], trainable_vars)

        _opts["generator"].apply_gradients(zip(grads, trainable_vars))
```
The same changes are applied to the discriminator training, too.

And the optimizer objects must be wrapped in a `LossScaleOptimizer()` class:
```python
    opts = {
        "generator" : tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.001)),
        "discriminator" : tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.RMSprop(0.0001))
    }
```

We have also added a few quality-of-life improvements to this script, including:
- A switch at the top of the file to activate `float16` mixed precision
- TensorBoard logging of discriminator and generator loss, per epoch

In the GAN, it shouldn’t make as big of a difference since the loss never gets so low that
you get near the clipping range, but it could matter.
![GAN loss scaling comparison](images/GAN-loss-curves-loss-scaling-vs-none-vs-float32.png)

So a GAN was not the best example to illustrate loss scaling. But certainly LSTMs are more
vulnerable to these concerns, as shown in this example from NVIDIA:
![bigLSTM loss scaling importance](images/NVIDIA-bigLSTM-loss-scaling.png)


## Manual profiling
If you grabbed a full DGX node, let's exclude all but one A100 GPU, since we are not
considering distributed training:
```
export CUDA_VISIBLE_DEVICES=0
```
By default, TensorFlow will allocate memory on every visible GPU, and the diagnostics for
the unused GPUs will annoyingly clog `STDOUT`. 

First, let's check the importance of TF32 mode.

With `use_mixed_precision=False` at the top of the file so that operands are all single
precision, execute the code with TF32 disabled at the system level:
```
Singularity> NVIDIA_TF32_OVERRIDE=0 python train_GAN_optimized.py
```
We recall that we achieved 90-100K img/s with TF32 enabled. The throughput drops by 33%
when none of the `float32` operands are able to utilize any of the 492 TCs:
```
2021-05-05 04:30:05,967 - INFO - (2, 13), G Loss: 0.699, D Loss: 0.612, step_time: 0.129, throughput: 63603.233 img/s.
2021-05-05 04:30:06,212 - INFO - (3, 0), G Loss: 0.698, D Loss: 0.615, step_time: 0.129, throughput: 63499.559 img/s.
2021-05-05 04:30:06,351 - INFO - (3, 1), G Loss: 0.700, D Loss: 0.615, step_time: 0.129, throughput: 63326.235 img/s.
2021-05-05 04:30:06,490 - INFO - (3, 2), G Loss: 0.698, D Loss: 0.618, step_time: 0.129, throughput: 63691.775 img/s.
```

Note, `NVIDIA_TF32_OVERRIDE=0` will have no effect in this example if
`use_mixed_precision=True`, since the Tensor cores can take `float16` inputs and output
`float16` matrices without using a TF32 intermediate format.




## NVIDIA DLProf

`nvidia-smi` has no way of showing Tensor Core utilization. Nor is it really exposed in
Nsight Systems. 

For deep learning software, we can use DLProf to profile TC utilization and more. 
This software lives on top of the Nsight Systems and Nsight Compute profilers. The results
can be dumped to CSV and/or visualized in TensorBoard. You can correlate GPU performance
with the model timeline. 

In TensorFlow 2.x, there is no universal marker for the beginning/end of a training
interation. Therefore, we must specify an operation ("node") which demarcates the
iteration boundary. Use the `--key_node` flag; [this
documention](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/#find_good_key_node)
describes a two-step process for determining a good node/op. 

Also, we want to exclude the first handful of training iterations from profiling
statistics, since they are always slower due to XLA compilation, cuDNN heuristics, etc.:

```
dlprof --key_node=ASSIGNADDVARIABLEOP_1 --iter_start=15 python train_GAN_optimized.py
```

Note, `dlprof` will not complete successfully if you send `SIGINT` to your application. I
have also found that `--iter_stop` caused issues in the environment / with this particular
version. If you are on a single GPU interactive job, the profiler might fail to detect
AMP. 

![float32 no TF32 profile](images/float32-XLA-disable-TF32-dlprof.png)

![float32 TF32 profile](images/float32-XLA-TF32-dlprof.png)

![float16 profile](images/float16-XLA-dlprof.png)

### Pitfall: incompatible tensor sizes

One of the most common mistakes when it comes to using Tensor Cores

<!-- Ensuring GPU Tensor Cores are used
As mentioned previously, modern NVIDIA GPUs use a special hardware unit called Tensor Cores that can multiply float16 matrices very quickly. However, Tensor Cores requires certain dimensions of tensors to be a multiple of 8. In the examples below, an argument is bold if and only if it needs to be a multiple of 8 for Tensor Cores to be used.

tf.keras.layers.Dense(units=64)
tf.keras.layers.Conv2d(filters=48, kernel_size=7, stride=3)
And similarly for other convolutional layers, such as tf.keras.layers.Conv3d
tf.keras.layers.LSTM(units=64)
And similar for other RNNs, such as tf.keras.layers.GRU
tf.keras.Model.fit(epochs=2, batch_size=128)
--> 


Let's mess up the generator network. First, we change the batch size from 4096 to 4090. 
Next, we change the number of filters in each convolutional and dense layer from a
multiple of 8 to an odd number




![float16 profile with bad layer sizes](images/float16-XLA-dlprof-bad-sizes.png)

![float16 profile with bad layer sizes- zoomed](images/float16-XLA-dlprof-bad-sizes-iter.png)
