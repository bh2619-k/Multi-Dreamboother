import argparse
import sys
from pathlib import Path

from dreambooth.data.data_utils import (
    directory_exists,
    directory_is_empty,
    remove_content_inside_dir,
)
from dreambooth.inference.dreambooth_infer import DreamboothFinetunedModelInference
from dreambooth.inference.pretrained_infer import PretrainedModelInference
from dreambooth.logger.logger_factory import LoggerFactory
from dreambooth.training.trainer import DreamBoothTrainer

LoggerFactory.configure()
log = LoggerFactory.get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="DreamBooth utility for Stable Diffusion - train or inference."
    )

    # Two modes: train, infer
    sub = parser.add_subparsers(dest="mode", required=True)

    # === TRAIN BRANCH ===
    train_p = sub.add_parser("train", help="Train a DreamBooth model")
    train_p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base HF model repo (e.g. stable-diffusion-v1-5/stable-diffusion-v1-5)",
    )
    train_p.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder with training images, collection of several images of a subject",
    )
    train_p.add_argument(
        "--save_folder", type=str, required=True, help="Where to save trained model"
    )
    train_p.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="In addition to diffusion model, also fine-tune text encoder",
    )
    train_p.add_argument(
        "--text_encoder_training_mode",
        type=str,
        default="textual_inversion",
        help=(
            "Specifies how the text encoder should be trained. "
            "Use 'textual_inversion' to train only a new embedding for a special "
            "token added to the tokenizer, or 'full' mode to train the "
            "entire text encoder."
        ),
    )
    train_p.add_argument(
        "--instance_prompts",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more prompts describing the subject using your unique identifier"
            " token (e.g., 'sks') and class family of subject (e.g., 'dog'). If a "
            "single prompt is provided, it is applied to all subject images. If "
            "multiple prompts are provided, they are matched to subject images in "
            "alphabetical filename order, and the number of prompts must equal"
            " the number of images. Each prompt must include the identifier token"
            " and class family of subject. "
            'Example: --instance_prompts "a photo of sks dog" or '
            '--instance_prompts "a sks dog in the jungle" "a sks dog at the beach"'
            ". [*] Using a single prompt is recommended since it is more stable "
            "unless you intentionally want per-image variation."
        ),
    )
    train_p.add_argument(
        "--special_token",
        type=str,
        required=True,
        help=(
            "The special token used in '--instance_prompts' (e.g., 'sks'). "
            "If you used a different token in the instance prompts, you must use "
            "the same token here and during inference."
        ),
    )
    train_p.add_argument(
        "--class_prompt",
        type=str,
        help="General class description without the unique token, "
        "e.g., 'a photo of a dog'. Used for prior-preservation during training.",
    )
    train_p.add_argument(
        "--num_worker",
        type=int,
        default=4,
        help=("Number of worker processes used by the DataLoader."),
    )
    train_p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "Mini batch size for fine-tuning stable diffusion. "
            "If prior preservation loss is on, it should be an even number."
        ),
    )
    train_p.add_argument("--max_steps", type=int, default=400, help="Training steps")
    train_p.add_argument(
        "--learning_rate",
        type=float,
        default=2.0e-06,
        help="Training learning rate for all trainable components except"
        " text encoder, e.g., 2e-6.",
    )
    train_p.add_argument(
        "--learning_rate_text_encoder",
        type=float,
        default=1.0e-06,
        help="Learning rate used when training the text encoder(s), e.g., 1e-6."
        " Only considered when train_text_encoder is true",
    )
    train_p.add_argument(
        "--prior_preservation",
        action="store_true",
        help=(
            "Turn on prior-preservation loss to keep the subject aligned with "
            " its class. When enabled, batches include both subject images and"
            " images generated from a frozen pre-tuning model using a general "
            "class prompt. Requires batch size >1. Typically used with "
            "--instance_prompts and --class_prompt."
        ),
    )
    train_p.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Weight applied to the prior-preservation loss when --prior_preservation "
            " is enabled Controls the balance between the instance loss and the class"
            " (prior) loss."
        ),
    )
    train_p.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Total number of class images to generate when using --prior_preservation. "
            "These images are used to compute the class (prior) loss."
        ),
    )
    train_p.add_argument(
        "--class_generation_batch_size",
        type=int,
        default=16,
        help=(
            "Batch size to use when generating class images for prior preservation. "
            "Lower values reduce VRAM usage but increase generation time."
        ),
    )
    train_p.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Classifier-free guidance scale used when generating class images with the "
            "frozen pretrained model for prior preservation. Higher values produce "
            "sharper, more class-consistent samples but may reduce diversity."
        ),
    )
    train_p.add_argument(
        "--infer_steps",
        type=int,
        default=80,
        help="Number of diffusion steps used to generate class images for prior "
        "preservation.",
    )
    train_p.add_argument(
        "--class_generation_dir",
        type=str,
        default="./frozen_model_generation_for_pr_prsv",
        help=(
            "Directory where generated class images will be stored when "
            "prior preservation is enabled."
        ),
    )
    train_p.add_argument(
        "--reuse_class_generation_dir",
        action="store_true",
        help=(
            "If the directory used to store generated class images already exists "
            "when prior preservation is enabled, reuse the existing images instead "
            "of generating new ones."
        ),
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducibility.",
    )

    # === INFER BRANCH ===
    infer_p = sub.add_parser("infer", help="Run inference with a trained model")
    infer_p.add_argument(
        "--load_path",
        type=str,
        required=True,
        help='Folder containing trained model (e.g. "./saved_model")',
    )
    infer_p.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        required=True,
        help='List of prompts (e.g. "a photo of sks dog in Mars" "an oil paining '
        'of sks dog in London")',
    )
    infer_p.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, watermark, text, jpeg artifacts",
        help="Negative prompt. This is shared for all inputed prompts.",
    )
    infer_p.add_argument(
        "--num_img_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each prompt. "
        "For example, with 3 prompts and --num_img_per_prompt 2, "
        "a total of 6 images will be produced.",
    )
    infer_p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Number of prompts to process at once during inference. "
            "Higher values use more GPU memory but speed up generation."
        ),
    )
    infer_p.add_argument(
        "--guidance_scale", type=float, default=7.5, help="CFG guidance scale"
    )
    infer_p.add_argument("--steps", type=int, default=80, help="Inference steps")
    infer_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducibility.",
    )
    infer_p.add_argument(
        "--output_sample_dir",
        type=str,
        default="./generated_samples",
        help="Directory path to store generated images.",
    )

    # Parse normally
    args, unknown = parser.parse_known_args()

    # Reject unknown flags entirely (optional strictness)
    if unknown:
        log.error(f"Unknown or misplaced arguments: {unknown}")
        sys.exit(1)

    # --- Train branch
    if args.mode == "train":

        input_dir = Path(args.input_folder)
        if not directory_exists(path=input_dir):
            raise NotADirectoryError(f"Input folder invalid: {input_dir}")
        if directory_is_empty(path=input_dir):
            raise NotADirectoryError(f"Input folder is empty: {input_dir}")

        save_dir = Path(args.save_folder)
        save_dir.mkdir(parents=True, exist_ok=True)
        remove_content_inside_dir(dir=save_dir)

        log.info("\n=== TRAIN MODE ===")
        log.info("Input arguments: {}".format(args))

        # --- If prior preservation is on, generate samples
        # with pretrained frozen model
        if args.prior_preservation:
            class_generation_dir = Path(args.class_generation_dir)
            if not args.reuse_class_generation_dir:
                # Create directory for saving generated images with
                # frozen pretrained model
                class_generation_dir.mkdir(parents=True, exist_ok=True)
                # Remove all contents inside the directory
                remove_content_inside_dir(dir=class_generation_dir)

                frozen_pretrained_model = PretrainedModelInference(
                    model_name=args.model_name,
                    prompts=[args.class_prompt] * args.num_class_images,
                    batch_size=args.class_generation_batch_size,
                    guidance_scale=args.guidance_scale,
                    steps=args.infer_steps,
                    seed=args.seed,
                )
                frozen_pretrained_model.load_pipe_for_inference()
                frozen_pretrained_model.generate_and_save_images(
                    save_folder=class_generation_dir
                )
            else:
                # In case of resuing previously generated images with
                # frozon model, the folder should exist
                if not directory_exists(path=class_generation_dir):
                    raise NotADirectoryError(
                        "Folder for generated images by frozen model"
                        f"invalid: {class_generation_dir}"
                    )
                if directory_is_empty(path=class_generation_dir):
                    raise NotADirectoryError(
                        "Folder for generated images by frozen model"
                        f" is empty: {class_generation_dir}"
                    )
                log.info(
                    "Previously generated images with pretrained"
                    f" frozen model in folder {class_generation_dir} "
                    f"will be used."
                )

        # Dreambooth trainer object
        trainer = DreamBoothTrainer(
            model_name=args.model_name,
            input_folder=args.input_folder,
            save_folder=args.save_folder,
            train_text_encoder=args.train_text_encoder,
            text_encoder_training_mode=args.text_encoder_training_mode,
            instance_prompts=args.instance_prompts,
            special_token=args.special_token,
            class_prompt=args.class_prompt,
            num_worker=args.num_worker,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            learning_rate_text_encoder=args.learning_rate_text_encoder,
            prior_preservation=args.prior_preservation,
            prior_loss_weight=args.prior_loss_weight,
            class_generation_dir=args.class_generation_dir,
            seed=args.seed,
        )
        trainer.train()

        return

    # --- Inferemce branch

    if args.mode == "infer":

        log.info("\n=== Inference MODE ===")
        log.info("Input arguments: {}".format(args))

        output_sample_dir = Path(args.output_sample_dir)
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        remove_content_inside_dir(dir=output_sample_dir)

        inference = DreamboothFinetunedModelInference(
            load_path=args.load_path,
            prompts=args.prompt,
            negative_prompt=args.negative_prompt,
            num_img_per_prompt=args.num_img_per_prompt,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed,
        )
        inference.load_pipe_for_inference()
        inference.generate_and_save_images(save_folder=output_sample_dir)


if __name__ == "__main__":
    main()
