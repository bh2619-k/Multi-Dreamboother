```
 dreambooth train \
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/dog \
    --save_folder ./saved-model/dog \
    --train_text_encoder \
    --text_encoder_training_mode full \
    --instance_prompts "a photo of <q5xv> dog" \
    --special_token "<q5xv>" \
    --class_prompt "a photo of a dog" \
    --num_worker 4 \
    --batch_size 2 \
    --max_steps 600 \
    --learning_rate 2.0e-06 \
    --learning_rate_text_encoder 2.0e-06 \
    --prior_preservation \
    --num_class_images 140 \
    --class_generation_batch_size 20 \
    --guidance_scale 7.5 \
    --infer_steps 60 \
    --class_generation_dir ./frozen_model_generation_for_pr_prsv/dog
```

```
 dreambooth infer \
    --load_path ./saved-model/dog \
    --prompt "a high-resolution photo of <q5xv> dog standing on the surface of Mars, dramatic red landscape, sharp details" \
             "a detailed oil painting of <q5xv> dog in London, Big Ben in the background, rich colors, artistic brush strokes" \
             "<q5xv> dog floating in outer space, stars and nebulae in the background, cinematic lighting" \
             "<q5xv> dog in Middle Earth, epic fantasy scenery, lush green valleys, Tolkien style" \
             "a photo of <q5xv> dog with blue hat" \
             "a metalic statue of <q5xv> dog" \
             "a photo of <q5xv> dog driving a car" \
             "a photo of <q5xv> dog with purple fur" \
             "a high-quality photo of <q5xv> dog skiing on a snowy mountain, sharp details, natural lighting" \
             "a Van Gogh painting of <q5xv> dog" \
    --negative_prompt "blurry, low quality, watermark, text, jpeg artifacts" \
    --num_img_per_prompt 10 \
    --batch_size 20 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images/dog
```
