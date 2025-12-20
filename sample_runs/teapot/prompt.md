```
 dreambooth train \
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/teapot \
    --save_folder ./saved-model/teapot \
    --instance_prompts "a photo of <q5xv> teapot" \
    --special_token "<q5xv>" \
    --class_prompt "a photo of a teapot" \
    --num_worker 4 \
    --batch_size 2 \
    --max_steps 800 \
    --learning_rate 2.0e-06 \
    --prior_preservation \
    --num_class_images 140 \
    --class_generation_batch_size 20 \
    --guidance_scale 7.5 \
    --infer_steps 60 \
    --class_generation_dir ./frozen_model_generation_for_pr_prsv/teapot
```

```
 dreambooth infer \
    --load_path ./saved-model/teapot \
    --prompt "a photo of <q5xv> teapot floating in milk." \
             "a moody studio photo of <q5xv> teapot sitting on a wet reflective surface, soft rain droplets, neon rim lighting, dark background, cinematic contrast, ultra-detailed, professional product photography, bokeh highlights" \
             "a photo of <q5xv> teapot with flower design." \
             "a <q5xv> teapot made of glass." \
             "a yellow <q5xv> teapot." \
             "a metallic <q5xv> teapot." \
             "a transparent <q5xv> teapot full of blue liquid." \
             "a <q5xv> teapot in library." \
             "a <q5xv> teapot floating in the sea." \
             "a <q5xv> teapot on fireplace." \
             "a photo of <q5xv> teapot in snow." \
             "a close-up of <q5xv> teapot with red strips." \
    --negative_prompt "blurry, low quality, watermark, text, jpeg artifacts" \
    --num_img_per_prompt 7 \
    --batch_size 20 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images/teapot
```
