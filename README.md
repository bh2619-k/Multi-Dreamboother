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
If you want to change the version of cuda enabled torch, you can modify install-gpu section in `Makefile`.

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
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/dog \
    --save_folder ./saved-model \
    --train_text_encoder \
    --instance_prompt "a photo of sks dog" \
    --class_prompt "a photo of a dog" \
    --num_worker 4 \
    --batch_size 2 \
    --max_steps 400 \
    --learning_rate 2.0e-06 \
    --prior_preservation \
    --num_class_images 200 \
    --class_generation_batch_size 20
```

### Inference

```
 dreambooth infer \
    --load_path ./saved-model \
    --prompt "a high-resolution photo of sks dog standing on the surface of Mars, dramatic red landscape, sharp details" \
             "a detailed oil painting of sks dog in London, Big Ben in the background, rich colors, artistic brush strokes" \
             "sks dog floating in outer space, stars and nebulae in the background, cinematic lighting" \
             "sks dog in Middle Earth, epic fantasy scenery, lush green valleys, Tolkien style" \
             "a sks dog with blue hat" \
    --negative_prompt "blurry, low quality, watermark, text, jpeg artifacts" \
    --num_img_per_prompt 4 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images
```

## Example Results


## License
