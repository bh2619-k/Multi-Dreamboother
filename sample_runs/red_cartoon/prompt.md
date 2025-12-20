```
 dreambooth train \
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/red_cartoon \
    --save_folder ./saved-model/red_cartoon \
    --train_text_encoder \
    --text_encoder_training_mode full \
    --instance_prompts "a photo of <q5xv> cartoon" \
    --special_token "<q5xv>" \
    --class_prompt "a photo of a cartoon" \
    --num_worker 4 \
    --batch_size 2 \
    --max_steps 800 \
    --learning_rate 1.0e-06 \
    --learning_rate_text_encoder 1.0e-06 \
    --prior_preservation \
    --num_class_images 140 \
    --class_generation_batch_size 20 \
    --guidance_scale 7.5 \
    --infer_steps 60 \
    --class_generation_dir ./frozen_model_generation_for_pr_prsv/red_cartoon
```

```
 dreambooth infer \
    --load_path ./saved-model/red_cartoon \
    --prompt "a photo of <q5xv> cartoon riding a bicycle." \
             "a photo of <q5xv> cartoon on top of mount Fuji" \
             "a photo of <q5xv> cartoon with mustache" \
             "a photo of <q5xv> cartoon sit on a wooden chair in a house on fire" \
             "a photo of blue <q5xv> cartoon" \
             "a photo of <q5xv> cartoon with long blond hair" \
             "a photo of <q5xv> cartoon in front of Eiffel tower." \
             "a photo of <q5xv> cartoon in bottom of ocean near coral cliff." \
             "a photo of <q5xv> cartoon in ramen bowl." \
             "a photo of <q5xv> cartoon with purple fur" \
             "an oil painting of <q5xv> cartoon" \
             "a <q5xv> cartoon dressed as Ninja" \
             "an <q5xv> cartoon with a sword" \
             "a photo of <q5xv> cartoon on Moon" \
             "a pixel art of <q5xv> cartoon" \
             "a photo of metalic statue of <q5xv> cartoon" \
             "a photo of <q5xv> cartoon at white house" \
             "a photo of <q5xv> cartoon at space" \
    --negative_prompt "blurry, low quality, watermark, text, jpeg artifacts" \
    --num_img_per_prompt 7 \
    --batch_size 20 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images/red_cartoon
```
