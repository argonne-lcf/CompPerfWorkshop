# DeepSpeed

From: [`Microsoft/DeepSpeedExamples`](https://github.com/microsoft/DeepSpeedExamples/blob/36212dd59c/HelloDeepSpeed/README.md)

> 📌 **Note:** 
> You can check out [[1](https://github.com/microsoft/DeepSpeedExamples/blob/36212dd59c/HelloDeepSpeed/README.md#1), [2](https://github.com/microsoft/DeepSpeedExamples/blob/36212dd59c/HelloDeepSpeed/README.md#2)] as a starting point for better understanding Transformers. Additionally, there are a number of blogs that do a nice deep dive into the workings of these models (eg: [this](https://nlp.seas.harvard.edu/2018/04/03/attention.html), [this](https://jalammar.github.io/illustrated-bert/) and [this](https://jalammar.github.io/illustrated-transformer/)).

```json
ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
        }
    },
}
```

```python
model, _, _, _ = deepspeed.initialize(model=model,
                                      config=ds_config,
                                      model_parameters=model.parameters())
```

This will create the DeepSpeed training engine based on the previously instantiated model and the new `ds_config` dictionary. We can now also remove the previous lines of code that created an Adam optimizer, this will now be done via the DeepSpeed engine. It should be noted, you can optionally created your own optimizer and pass it into `deepspeed.initialize` however DeepSpeed is able to make further performance optimizations by instantiating its own optimizers.

### Update the training-loop

Next we want to update our training-loop to use the new model engine with the following changes:

-   `model.to(device)` can be removed
    -   DeepSpeed will be careful on when to move the model to GPU to reduce GPU memory usage (e.g., converts to half on CPU then moves to GPU)
-   `optimizer.zero_grad()` can be removed
    -   DeepSpeed will do this for you at the right time.
-   Replace `loss.backward()` with `model.backward(loss)`
    -   There are several cases where the engine will properly scale the loss when using certain features (e.g., fp16, gradient-accumulation).
-   Replace `optimizer.step()` with `model.step()`
    -   The optimizer step is handled by the engine now and is responsible for dispatching to the right optimizer depending on certain features.