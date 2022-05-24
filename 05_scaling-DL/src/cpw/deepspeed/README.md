# Distributed training with Microsoft DeepSpeed

Presented by [Sam Foreman](https://samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov)) @ ALCF Computational Performance, 2022
Modified from: [argonne-lcf/sdl_ai_workshop/01_distributedDeepLearning/README.md](https://github.com/argonne-lcf/sdl_ai_workshop/blob/master/01_distributedDeepLearning/DeepSpeed/README.md), by Zhen Xie ([zhen.xie@anl.gov](mailto:///zhen.xie@anl.gov))

[DeepSpeed](https://www.deepspeed.ai/) is a deep learning optimization library that makes distributed training easy, efficient, and effective.

Additional information can be found either in their github repository available at  [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) , or in their paper[^1], [^2], [^3]

> 💡 [DeepSpeed](http://www.deepspeed.ai) is a deep learning optimization library that makes distributed training easy, efficient, and effective. 
> <br>_**10x Larger Models**_ | _**10x Faster Training**_ | _**Minimal Code Change**_ 
> <br>DeepSpeed can train DL models with over a hundred billion parameters on current generation of GPU clusters, while achieving over 10x in system performance compared to the state-of-art.
> Early adopters of DeepSpeed have already produced a language model (LM) with over 17B parameters called [Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft), establishing a new SOTA in the LM category. 
> <br>DeepSpeed is an important part of Microsoft’s new [AI at Scale](https://www.microsoft.com/en-us/research/project/ai-at-scale/) initiative to enable next-generation AI capabilities at scale. Take a deep dive into [large scale AI across Microsoft](https://innovation.microsoft.com/en-us/exploring-ai-at-scale) 

- [🤗 Transformers](https://www.github.com/huggingface/transformers) also have an excellent writeup in their documentation on integrating DeepSpeed.
  - [📃 documentation](https://huggingface.co/docs/transformers/main_classes/deepspeed)
  - [💻 github](https://github.com/huggingface/transformers/blob/3f936df66287f557c6528912a9a68d7850913b9b/src/transformers/deepspeed.py)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed) also has a good reference + examples on DeepSpeed integration.


## Getting Started
Microsoft's [Getting Started - DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) page has a good high-level overview of the DeepSpeed engine.

We include some of the major points here:
  - Installing is as simple as pip install deepspeed, see more details.
  - DeepSpeed has direct integrations with HuggingFace Transformers and PyTorch Lightning.
    - HuggingFace Transformers users can now easily accelerate their models with DeepSpeed through a simple `--deepspeed` flag + config file [see more details](https://huggingface.co/transformers/main_classes/trainer.html#deepspeed)
    - PyTorch Lightning provides easy access to DeepSpeed through the Lightning Trainer [see more details](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed)

## How does DeepSpeed work?

- DeepSpeed delivers extreme-scale model training for everyone, from data scientists training on massive supercomputers to those training on low-end clusters or even on a single GPU:
  - Extreme scale: Using current generation of GPU clusters with hundreds of devices, 3D parallelism of DeepSpeed can efficiently train deep learning models with trillions of parameters.
  - Extremely memory efficient: With just a single GPU, ZeRO-Offload of DeepSpeed can train models with over 10B parameters, 10x bigger than the state of arts, democratizing multi-billion-parameter model training such that many deep learning scientists can explore bigger and better models.
- Extremely long sequence length: Sparse attention of DeepSpeed powers an order-of-magnitude longer input sequence and obtains up to 6x faster execution comparing with dense transformers.  
- Extremely communication efficient: 3D parallelism improves communication efficiency allows users to train multi-billion-parameter models 2–7x faster on clusters with limited network bandwidth.  1-bit Adam/1-bit LAMB reduce communication volume by up to 5x while achieving similar convergence efficiency to Adam/LAMB, allowing for scaling to different types of GPU clusters and networks. 
- Early adopters of DeepSpeed have already produced a language model (LM) with over 17B parameters called [Turing-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft), establishing a new SOTA in the LM category.

## Setting up DeepSpeed

The quickest way to get started with DeepSpeed is via pip, this will install the latest release of DeepSpeed which is not tied to specific PyTorch or CUDA versions.

DeepSpeed includes several C++/CUDA extensions that we commonly refer to as our 'ops'.  By default, all of these extensions/ops will be built just-in-time (JIT) using [torch's JIT C++ extension loader that relies on ninja](https://pytorch.org/docs/stable/cpp_extension.html) to build and dynamically link them at runtime.

**Note:** [PyTorch](https://pytorch.org/) must be installed _before_ installing DeepSpeed.

```bash
pip install deepspeed
```

After installation, you can validate your install and see which extensions/ops your machine is compatible with via the DeepSpeed environment report.

```bash
ds_report
```

If you would like to pre-install any of the DeepSpeed extensions/ops (instead of JIT compiling) or install pre-compiled ops via PyPI please see our [advanced installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/)

## Required Code Modifications
### Update the Training Loop
From: [`Microsoft/DeepSpeedExamples`](https://github.com/microsoft/DeepSpeedExamples/blob/36212dd59c/HelloDeepSpeed/README.md)

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



## Example

There is an example deployed by DeepSpeed here, with PyTorch using CIFAR-10.

For interactive job: 

It has two steps:

### Step 1: "Setting up env"
```bash
conda env create --name deepspeed --file /lus/theta-fs0/projects/datascience/zhen/env_deepspeed.yml //set up env and install packages
```
```bash
conda activate deepspeed // activate env
```

### Step 2: "Run script"
```bash
cd /lus/theta-fs0/projects/datascience/zhen/DeepSpeed
```
```bash
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@
```

For submitting jobs in the script (non-interactive) job mode, take a look in the `submissions/` folder for more details about this.


## DeepSpeed Performance

* [DeepSpeed powers 8x larger MoE model training with high performance](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/)
  * [Mixture of Experts (MoE) tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/).
* [Curriculum learning: a regularization method for stable and 2.6x faster GPT-2 pre-training with 8x/4x larger batch size/learning rate](https://www.deepspeed.ai/tutorials/curriculum-learning/)
* [DeepSpeed: Accelerating large-scale model inference and training via system optimizations and compression](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)
* [1-bit LAMB: up to 4.6x less communication and 2.8x faster training, together with LAMB's convergence speed at large batch sizes](https://www.deepspeed.ai/tutorials/onebit-lamb/)
* [ZeRO-Infinity unlocks unprecedented model scale for deep learning training](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)
  * [Tutorial on how to use different stages of ZeRO](https://www.deepspeed.ai/tutorials/zero/)
* [[DeepSpeed on AzureML] Transformers and CIFAR examples are now available on AzureML GitHub](https://github.com/Azure/azureml-examples/tree/main/python-sdk/workflows/train/deepspeed)
* [[PyTorch Lightningging Face Blog] Fit More and Train Faster With ZeRO via DeepSpeed and FairScale](https://huggingface.co/blog/zero-deepspeed-fairscale)

[^1]: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. (2019) ZeRO: memory optimizations toward training trillion parameter models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) and [In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)](https://dl.acm.org/doi/10.5555/3433701.3433727).
[^2]: Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. (2020) DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. [In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20, Tutorial)](https://dl.acm.org/doi/10.1145/3394486.3406703).
[^3]: Minjia Zhang, Yuxiong He. (2020) Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping. [arXiv:2010.13369](https://arxiv.org/abs/2010.13369) and [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html).
