```
 dreambooth train \
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/monster_toy \
    --save_folder ./saved-model/monster_toy \
    --train_text_encoder \
    --text_encoder_training_mode full \
    --instance_prompts "a photo of <q5xv> toy" \
    --special_token "<q5xv>" \
    --class_prompt "a photo of a toy" \
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
    --class_generation_dir ./frozen_model_generation_for_pr_prsv/monster_toy
```

```
 dreambooth infer \
    --load_path ./saved-model/monster_toy \
    --prompt "a photo of <q5xv> toy riding a bicycle." \
             "a photo of <q5xv> toy on top of mount Fuji" \
             "a photo of <q5xv> toy in front of Eiffel tower." \
             "a photo of <q5xv> toy in ocean." \
             "a photo of <q5xv> toy in ramen bowl." \
             "a photo of <q5xv> toy with purple fur" \
             "an oil painting of <q5xv> toy" \
             "a <q5xv> toy dressed as Ninja" \
             "an <q5xv> toy with a sword" \
             "a photo of <q5xv> toy on Moon" \
             "a pixel art of <q5xv> toy" \
             "a photo of metalic statue of <q5xv> toy" \
             "a photo of <q5xv> toy at white house" \
             "a photo of <q5xv> toy at space" \
    --negative_prompt "blurry, low quality, watermark, text, jpeg artifacts" \
    --num_img_per_prompt 7 \
    --batch_size 20 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images/monster_toy
```