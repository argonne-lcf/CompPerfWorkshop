# Auotmatic Mixed Precision in PyTorch

-----------
Training in FP16 that is in half precision results in slightly faster training in nVidia cards that supports half precision ops. Also the memory requirements of the models weights are almost halved since we use 16-bit format to store the weights instead of 32-bits.

Native support in `torch.cuda.amp` as of [PyTorch
1.6](https://pytorch.org/blog/pytorch-1.6-released/) (July 2020) and NGC container 20.6.

The procedure is very similar to that described in [TensorFlow](../TensorFlow/README.md):

1. Instantiate automated loss scaling
```
# Creates once at the beginning of training 
scaler = torch.cuda.amp.GradScaler()
```
2. Within a training iteration, wrap the forward pass in a way that casts the
compute-bound operations to mixed precision:
```
with torch.cuda.amp.autocast():
    loss = model(data)
```
3. Scale the loss and compute scaled gradients during backward pass:
```
scaler.scale(loss).backward()
```
4. Unscale gradients and update trainable weights:
```
scaler.step(optimizer)
```
5. Update the loss scaling factor
```
scaler.update()
```

Essentially, the only difference is: instead of wrapping the optimizer object with a loss
scaling optimizer object, PyTorch has the user explicitly call the adaptive loss scale
update from a separate object.


We will use the 2022 release NVIDIA-optimiized NGC Singularity container for
PyTorch 3.x in this walkthrough:
```
singularity exec --nv -B /lus /lus/theta-fs0/software/thetagpu/nvidia-containers/pytorch/pytorch_22.01-py3.simg bash
```


#### Usage Instruction
```
python main.py [-h] [--lr LR] [--steps STEPS] [--gpu] [--fp16] [--loss_scaling] [--model MODEL]

PyTorch (FP16) CIFAR10 Training

optional arguments:
  -h, --help            Show this help message and exit
  --lr LR               Learning Rate
  --steps STEPS, -n STEPS
                        No of Steps
  --gpu, -p             Train on GPU
  --fp16                Train with FP16 weights
  --loss_scaling, -s    Scale FP16 losses
  --model MODEL, -m MODEL
                        Name of Network
```
To run in `FP32` mode, use:  
`python main.py -n 200 -p --model resnet50`

To train with `FP16` weights, use:  
`python main.py -n 200 -p --fp16 -s --model resnet50`  
`-s` flag enables loss scaling.
