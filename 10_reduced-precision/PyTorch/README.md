# Auotmatic Mixed Precision in PyTorch

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

Refer to the [PyTorch AMP
Recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) for a detailed walkthrough.
