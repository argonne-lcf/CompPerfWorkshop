# Profiling Tensorflow

In this example, we'll profile a Generative network.  We'll go through several steps of profile, each time enabling a new tool or optimization.

Find the original script in `train_GAN.py`.

All the scripts used here work in the Tensorflow 2 container:

```bash
$ singularity exec --nv -B /lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_21.02-py3.simg bash
```


## A Starting Point

To download the mnist dataset, make sure to enable http forwarding:
```bash
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128
```

Run the original script, single node, like so:
`python train_GAN.py`

Take note of the throughput reported!

```
2021-04-28 20:05:54,202 - INFO - G Loss: 0.668, D Loss: 0.714, step_time: 1.016, throughput: 125.988 img/s.
2021-04-28 20:05:54,859 - INFO - G Loss: 0.822, D Loss: 0.641, step_time: 0.657, throughput: 194.946 img/s.
2021-04-28 20:05:55,452 - INFO - G Loss: 0.845, D Loss: 0.603, step_time: 0.593, throughput: 215.793 img/s.
2021-04-28 20:05:56,065 - INFO - G Loss: 0.916, D Loss: 0.577, step_time: 0.613, throughput: 208.956 img/s.
2021-04-28 20:05:56,647 - INFO - G Loss: 0.934, D Loss: 0.560, step_time: 0.582, throughput: 219.899 img/s.
2021-04-28 20:05:57,214 - INFO - G Loss: 0.978, D Loss: 0.543, step_time: 0.566, throughput: 226.013 img/s.
2021-04-28 20:05:57,769 - INFO - G Loss: 0.986, D Loss: 0.541, step_time: 0.555, throughput: 230.645 img/s.
2021-04-28 20:05:58,377 - INFO - G Loss: 1.023, D Loss: 0.548, step_time: 0.607, throughput: 210.964 img/s.
2021-04-28 20:05:58,948 - INFO - G Loss: 0.909, D Loss: 0.545, step_time: 0.571, throughput: 224.301 img/s.
2021-04-28 20:05:59,555 - INFO - G Loss: 0.902, D Loss: 0.563, step_time: 0.606, throughput: 211.146 img/s.
2021-04-28 20:06:00,150 - INFO - G Loss: 0.857, D Loss: 0.584, step_time: 0.582, throughput: 219.836 img/s.
2021-04-28 20:06:00,704 - INFO - G Loss: 0.760, D Loss: 0.577, step_time: 0.554, throughput: 231.169 img/s.
2021-04-28 20:06:01,268 - INFO - G Loss: 0.752, D Loss: 0.576, step_time: 0.564, throughput: 227.051 img/s.
2021-04-28 20:06:01,829 - INFO - G Loss: 0.734, D Loss: 0.587, step_time: 0.561, throughput: 228.336 img/s.
2021-04-28 20:06:02,380 - INFO - G Loss: 0.718, D Loss: 0.587, step_time: 0.551, throughput: 232.357 img/s.
2021-04-28 20:06:02,929 - INFO - G Loss: 0.692, D Loss: 0.605, step_time: 0.549, throughput: 233.260 img/s.
2021-04-28 20:06:03,479 - INFO - G Loss: 0.682, D Loss: 0.616, step_time: 0.550, throughput: 232.808 img/s.
2021-04-28 20:06:04,040 - INFO - G Loss: 0.700, D Loss: 0.623, step_time: 0.560, throughput: 228.521 img/s.
2021-04-28 20:06:04,603 - INFO - G Loss: 0.670, D Loss: 0.623, step_time: 0.563, throughput: 227.163 img/s.
2021-04-28 20:06:05,160 - INFO - G Loss: 0.682, D Loss: 0.616, step_time: 0.556, throughput: 230.337 img/s.
```

On average, the A100 system is moving about 230 Images / second through this training loop.  Let's dig in to the first optimization in the `line_profiler` directory.
