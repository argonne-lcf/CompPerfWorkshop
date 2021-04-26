## Hands on for Data Parallel Deep Learning on Theta (CPU)

1. SSH to Theta and request an interactive session on Theta from the Theta login node:

   ```bash
   $ ssh user@theta.alcf.anl.gov
   user@thetalogin4:~> qsub -n 4 -q training -A comp_perf_workshop -I -t 1:00:00
   ```

2. Setup the Python environment to include `TensorFlow, Keras, PyTorch, Horovod`

   ```bash
   user@thetamom1:~> module load datascience/pytorch-1.7
   user@thetamom1:~> module load datascience/tensorflow-2.3
   ```

3. Run the examples

   - Pytorch MNIST

     ```bash
     aprun -n 16 -N 4 -d 32 -j 2 -cc depth \
         -e OMP_NUM_THREADS=32 -e KMP_BLOCKTIME=0 \
         python pytorch_mnist.py --device cpu
     ```

   - TensorFlow MNIST

     ```bash
     aprun -n 16 -N 4 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth python tensorflow2_mnist.py --device cpu
     ```

4. Test scaling and investigate the issue of large batch size training. 

   **Note:** This requires a new job allocation to a separate job queue.

   The following script performs a simple scaling test with the MNIST dataset

   - PyTorch Model -- [submissions/theta/qsub_pytorch_mnist_scale.sh](./submissions/theta/qsub_pytorch_mnist_scale.sh):

     ```bash
     qsub -O pytorch_mnist_scale -n 128 -q training -A comp_perf_workshop submissions/theta/qsub_pytorch_mnist_scale.sh
     ```

   - TensorFlow with Keras API

     ```bash
     qsub -O tensorflow_mnist_scale -n 128 -q training -A comp_perf_workshop submissions/theta/qsub_keras_mnist_scale.sh
     ```

   You can check the test accuracy and timing for different scales.

   In this case, we run for 32 epochs. The time to solution decreases linearly as we increase the number of processes. The communication overhead is not significant.

   ![pytorch_mnist_time](../assets/theta_pytorch_mnist_time.png)



However, the training accuracy and test accuracy decreases as we increase the number of workers.

As we can see up to 128 processes, the training accuracy and test accuracy is ~10%, which means that the training does not converge to a local minimum at all. The reason is that when we scale the learning rate as `128 * args.lr_init = 128 * 0.01 = 1.28` which might be too large initially.

One solution is to slowly "warm up" the learning rate over multiple epochs at the beginning.

![theta_pytorch_mnist_acc](../assets/theta_pytorch_mnist_acc.png)

