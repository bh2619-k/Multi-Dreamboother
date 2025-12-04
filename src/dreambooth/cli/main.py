import argparse
import sys
from pathlib import Path

from dreambooth.data.data_utils import save_images
from dreambooth.inference.dreambooth_infer import DreamboothFinetunedModelInference
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
        "--instance_prompt",
        type=str,
        required=True,
        help="Describe the training subject using your special token + subject type, "
        "e.g., 'a photo of sks dog' or 'a photo of sks woman'. Keep the token "
        "next to the subject it refers to.",
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
        help="Training learning rate, e.g., 2e-6.",
    )
    train_p.add_argument(
        "--prior_preservation",
        action="store_true",
        help=(
            "Turn on prior-preservation loss to keep the subject aligned with "
            " its class. When enabled, batches include both subject images and"
            " images generated from a frozen pre-tuning model using a general "
            "class prompt. Requires batch size >1. Typically used with "
            "--instance_prompt and --class_prompt."
        ),
    )
    train_p.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help="Number of class images to generate when using --prior_preservation "
        "(used for class loss).",
    )
    train_p.add_argument(
        "--class_generation_batch_size",
        type=int,
        default=4,
        help="Batch size used when generating class images for prior preservation.",
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
        save_dir = Path(args.save_folder)

        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input folder invalid: {input_dir}")

        save_dir.mkdir(parents=True, exist_ok=True)

        log.info("\n=== TRAIN MODE ===")
        log.info("Input arguments: {}".format(args))

        # Dreambooth trainer object
        trainer = DreamBoothTrainer(
            model_name=args.model_name,
            input_folder=args.input_folder,
            save_folder=args.save_folder,
            train_text_encoder=args.train_text_encoder,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt,
            num_worker=args.num_worker,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            prior_preservation=args.prior_preservation,
            num_class_images=args.num_class_images,
            class_generation_batch_size=args.class_generation_batch_size,
            seed=args.seed,
        )
        trainer.train()

        return

    # --- Inferemce branch

    if args.mode == "infer":
        model_dir = Path(args.load_path)

        if not model_dir.is_dir():
            raise NotADirectoryError(f"Model load path invalid: {model_dir}")

        log.info("\n=== Inference MODE ===")
        log.info("Input arguments: {}".format(args))

        inference = DreamboothFinetunedModelInference(
            load_path=args.load_path,
            prompts=args.prompt,
            negative_prompt=args.negative_prompt,
            num_img_per_prompt=args.num_img_per_prompt,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed,
        )
        inference.load_pipe_for_inference()
        images = inference.generate_images()

        # Save generated images
        output_sample_dir = Path(args.output_sample_dir)
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = save_images(images, output_sample_dir, ext="png", overwrite=True)
        print(f"[save_images] Saved {len(saved_paths)} image(s) to {output_sample_dir}")


if __name__ == "__main__":
    main()
