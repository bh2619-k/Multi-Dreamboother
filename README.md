# DreamBooth

## Introduction
This repository provides a clean implementation of DreamBooth, a technique for fine-tuning text-to-image diffusion models (e.g., Stable Diffusion) on a specific subject using only a few reference images.
It is designed for easy training with a simple CLI (dreambooth train) and inference (dreambooth infer). The allowing you to plug in your own datasets, prompts, and training settings.

## Quick Installation

### 1. Create a virtual environment

``` bash
make venv
```

### 2. Activate it

``` bash
source .venv/bin/activate
```

### 3. Install Dependencies and CUDA-Enabled PyTorch

``` bash
make install-gpu
make install-dev
```
If you want to change the version of cuda enabled torch (currently CUDA 12.8), you can modify install-gpu section in `Makefile`.

### 4. Verify GPU support

``` bash
make check-gpu
```

## How To Use

Check supported commands and their options:
```
dreambooth --help
dreambooth train --help
dreambooth infer --help
```

For **examples** of how to use ```dreambooth train``` and ```dreambooth infer```, as well as some generated images, please see the ***```sample``` folder***.

## Example Results


## License
Released under the MIT License. For pretrained models or datasets, please check their respective licenses.