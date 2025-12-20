```
 dreambooth train \
    --model_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --input_folder ./input-data/google-dataset/face1 \
    --save_folder ./saved-model/face1 \
    --train_text_encoder \
    --text_encoder_training_mode full \
    --instance_prompts "a photo of <q5xv> person" \
    --special_token "<q5xv>" \
    --class_prompt "a photo of a person" \
    --num_worker 4 \
    --batch_size 2 \
    --max_steps 1000 \
    --learning_rate 1.0e-06 \
    --learning_rate_text_encoder 1.0e-06 \
    --prior_preservation \
    --num_class_images 200 \
    --class_generation_batch_size 20 \
    --guidance_scale 7.5 \
    --infer_steps 60 \
    --class_generation_dir ./frozen_model_generation_for_pr_prsv/face1
```

```
 dreambooth infer \
    --load_path ./saved-model/face1 \
    --prompt "a detailed oil painting of <q5xv> person, rich colors, artistic brush strokes." \
             "a photo of <q5xv> person as a wizard in Harry Potter Universe." \
             "a cinematic portrait of <q5xv> person illuminated by warm window light, shallow depth of field, expressive eyes, natural skin texture, ultra-detailed face, soft shadows, 85mm photography, professional color grading, high realism" \
             "a photo of <q5xv> person with makeup of the joker from the Dark Knight movie." \
             "a photo of <q5xv> person dressed as a roman emperor." \
             "a photo of <q5xv> person dressed as Aragorn form lord of the ring universe." \
             "a photo of <q5xv> person eating sandwich." \
             "a photo of <q5xv> person swimming near Coral reef." \
             "a photo of <q5xv> person as a hobbit in Shire." \
             "a photo of <q5xv> person hobbit." \
             "a portrait of <q5xv> person with green hair." \
             "a portrait of <q5xv> person with a pointy wizarding hat and dark cloack." \
             "A professional black and white studio portrait of <q5xv> person." \
             "A portrait of <q5xv> person inside a vintage cafe." \
             "A fantasy portrait of <q5xv> person as a powerful wizard." \
             "A medieval painting of <q5xv> person in knight armor" \
             "A Japanese-style painting of <q5xv> person as a samurai" \
    --negative_prompt "blurry, low quality, distorted, deformed face, extra limbs, watermark, text, cropped, jpeg artifacts,  bad anatomy, low resolution" \
    --num_img_per_prompt 10 \
    --batch_size 20 \
    --guidance_scale 7.5 \
    --steps 80 \
    --seed 42 \
    --output_sample_dir ./generated_images/face1
```