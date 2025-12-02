# DreamBooth

## Introduction

## Quick Installation

### 1. Create a virtual environment

``` bash
make venv
```

### 2. Activate it

``` bash
source .venv/bin/activate
```

### 3. Install GPU version (CUDA 12.8)

``` bash
make install-gpu
make install-dev
```

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

### Train

```
 dreambooth train \
    --model_name stabilityai/stable-diffusion-2-1 \
    --input_folder ./input-data/google-dataset/dog \
    --save_folder ./saved-model \
    --train_text_encoder \
    --instance_prompt "a photo of sks dog" \
    --class_prompt "a photo of a dog" \
    --batch_size 2 \
    --max_steps 400 \
    --learning_rate 2.0e-06 \
    --prior_preservation
```

### Inference

## Example Results

## License
