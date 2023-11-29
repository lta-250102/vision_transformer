<div align="center">

# Vision Transformer 

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper here](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2010.11929)

</div>

## Description

training Vision Transformer for image classification on several datasets (MNIST, CIFAR10, CIFAR100, ImageNet)

## Installation

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt
```

## How to train

#### Configure your experiment

Data configuration is located in [configs/data/vit](configs/data/vit.yaml)

```yaml
batch_size: int
dataset_to_down: str in (MNIST, CIFAR10, CIFAR100, ImageNet)
image_size: [int, int]
```

Model configuration is located in [configs/model/vit](configs/model/vit.yaml)

```yaml
learning_rate: float
nhead: int
dim_feedforward: int
blocks: int
mlp_head_units: [int, int] # n_feature in mlp
n_classes: int # base on dataset
img_size: [int, int] # base on dataset
patch_size: [int, int] # base on dataset
n_channels: 3 # with mnist is 1
d_model: int
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
