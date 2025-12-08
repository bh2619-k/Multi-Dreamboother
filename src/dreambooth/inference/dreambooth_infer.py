from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel

from dreambooth.data.data_utils import (
    directory_exists,
    directory_is_empty,
    save_image_batch,
)
from dreambooth.logger.logger_factory import LoggerFactory

log = LoggerFactory.get_logger()


class DreamboothFinetunedModelInference:
    """Common utilities and lifecycle for inference."""

    def __init__(
        self,
        load_path: str,
        prompts: list[str],
        negative_prompt: str,
        num_img_per_prompt: int,
        batch_size: int,
        guidance_scale: float,
        steps: int,
        seed: int,
    ):

        self.load_path = Path(load_path)
        self.prompts = prompts
        self.batch_size = batch_size
        self.negative_prompt = negative_prompt
        self.num_img_per_prompt = num_img_per_prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed

        self._validate_input_arguments()

        # HF Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Set seed for reproducebility
        self._setup_seed(self.seed)

        # Attributes of fine-tuned model with dreambooth training
        self.trained_model_attrs: dict = {}

        # Stable Diffusion pipeline
        self.pipe: StableDiffusionPipeline | None = None

    # --------------------------- Load model -----------------------------------

    def load_pipe_for_inference(self) -> StableDiffusionPipeline:
        """
        Load model into a StableDiffusionPipeline.
        """
        export_dir = self.load_path

        # Load the metadata dictionary
        meta = torch.load(export_dir / "meta.pt", map_location="cpu")
        self.trained_model_attrs["training_method"] = meta["training_method"]
        self.trained_model_attrs["model_name"] = meta["model_name"]
        self.trained_model_attrs["train_text_encoder"] = meta["train_text_encoder"]
        self.trained_model_attrs["global_step"] = meta["global_step"]
        self.trained_model_attrs["instance_prompt"] = meta["instance_prompt"]
        self.trained_model_attrs["prior_preservation"] = meta["prior_preservation"]
        self.trained_model_attrs["resolution"] = meta["resolution"]

        log.info(f"\nSaved model meta data: {self.trained_model_attrs}\n")

        # Build fine-tuned submodules first (only if dirs exist)
        unet_path = export_dir / "unet"
        if not unet_path.is_dir():
            raise FileNotFoundError(f"Missing UNet folder: {unet_path}")
        ft_unet = UNet2DConditionModel.from_pretrained(unet_path)

        te_path = export_dir / "text_encoder"
        if self.trained_model_attrs["train_text_encoder"] and not te_path.is_dir():
            raise RuntimeError(
                "Model was trained with text encoder, " "but no text encoder found."
            )
        ft_te = None
        if te_path.is_dir():
            ft_te = CLIPTextModel.from_pretrained(te_path)

        # Build pipeline with overrides so Accelerate can shard them
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "unet": ft_unet,
        }

        # Only override text_encoder if a finetuned one exists
        if ft_te is not None:
            pipe_kwargs["text_encoder"] = ft_te

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.trained_model_attrs["model_name"], **pipe_kwargs
        )
        # keep the whole pipe on local device.
        self.pipe.to(self.device)

        # Swap scheduler after construction
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Memory/perf niceties (safe to call even with device_map)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def get_pipe(
        self,
    ) -> StableDiffusionPipeline:
        if self.pipe is None:
            raise RuntimeError("Pipeline has not been created yet.")
        return self.pipe

    # --------------------------- Generation -----------------------------------
    def generate_image_batches(self) -> Iterator[list[Image.Image]]:
        """Yield batches of generated images using the loaded diffusion pipeline.

        Returns:
            Iterator[list[Image.Image]]: Each iteration yields a list of PIL images.
        """
        pipe = self.get_pipe()

        # Build the full prompt lists
        all_prompts = []
        for promt_i in self.prompts:
            all_prompts += [promt_i] * self.num_img_per_prompt
        all_neg_prompts = [self.negative_prompt] * len(all_prompts)

        steps = self.steps
        guidance = self.guidance_scale
        height, width = self.trained_model_attrs["resolution"]

        batch_size = self.batch_size
        total_prompts = len(all_prompts)
        number_of_batches = math.ceil(total_prompts / batch_size)

        for batch_i in range(number_of_batches):
            with torch.inference_mode():

                start = batch_i * batch_size
                end = min(start + batch_size, total_prompts)

                batch_prompts = all_prompts[start:end]
                batch_neg_prompts = all_neg_prompts[start:end]

                out = pipe(
                    prompt=batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    num_images_per_prompt=1,
                    output_type="pil",
                )

                images = out["images"] if isinstance(out, dict) else out.images
                if not images:
                    raise RuntimeError("Pipeline returned no images")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield images

    def generate_and_save_images(self, save_folder: Path) -> None:
        """
        Generate images in batches and save them to disk,
        logging progress every 20 images.
        """
        if not directory_exists(path=save_folder):
            raise NotADirectoryError(f"Save folder invalid: {save_folder}")
        if not directory_is_empty(path=save_folder):
            raise ValueError(f"Save folder should be empty: {save_folder}")

        total_generated = 0

        for images in self.generate_image_batches():
            images_dic = {
                f"image_{idx+total_generated}": image_i
                for idx, image_i in enumerate(images)
            }
            saved_paths = save_image_batch(
                images_dic, out_dir=save_folder, ext="png", overwrite=False
            )
            total_generated += len(saved_paths)

            # Log every 20 images
            if total_generated % 20 == 0:
                log.info(f"{total_generated} images generated and saved so far.")

        log.info(f"Generation complete. Total images saved: {total_generated}")

    # --------------------------- Other -----------------------------------

    def _setup_seed(self, seed: int, deterministic: bool = False) -> None:
        """Seed Python, NumPy, and Torch. Optionally make cuDNN deterministic."""
        # ensure full coverage
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        set_seed(
            seed, device_specific=True
        )  # different local_rank -> different seed offset
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _validate_input_arguments(
        self,
    ):
        if not directory_exists(path=self.load_path):
            raise NotADirectoryError(
                f"Folder to load saved model is invalid: {self.load_path}"
            )
        if directory_is_empty(path=self.load_path):
            raise NotADirectoryError(
                f"Folder to load saved model is empty: {self.load_path}"
            )
        if not isinstance(self.prompts, list):
            raise ValueError(
                f"Input prompts should be a list. Current value: {self.prompts}"
            )
        if len(self.prompts) == 0:
            raise ValueError(
                "Input prompts should have at least one element. "
                f"Current value: {self.prompts}"
            )
        for str_i in self.prompts:
            if not isinstance(str_i, str) or len(str_i) == 0:
                raise ValueError(
                    "Input prompts should be a non-empty valid strings. "
                    f"Current value: {self.prompts}, "
                    f"{str_i} violates the condition."
                )
        if not isinstance(self.negative_prompt, str) or len(self.negative_prompt) == 0:
            raise ValueError(
                "Negative prompt should be a non-empty valid strings. "
                f"Current value: {self.negative_prompt}, "
            )
        if not (isinstance(self.batch_size, int) and self.batch_size >= 1):
            raise ValueError(
                "Inference batch size should be an integer greater than 0. "
                f"Current batch_size = {self.batch_size}."
            )
        if not isinstance(self.num_img_per_prompt, int) or not (
            1 < self.num_img_per_prompt <= 100
        ):
            raise ValueError(
                "Number of generated images per prompt should be an int "
                "number between 0 and 100. "
                f"Current value is {self.num_img_per_prompt}"
            )
        if not isinstance(self.guidance_scale, (int, float)) or not (
            0 <= self.guidance_scale <= 10
        ):
            raise ValueError(
                "Guidance scale should be a float number between 0 and 10. "
                f"Current value is {self.guidance_scale}"
            )

        if not (isinstance(self.steps, int) and (1 <= self.steps <= 1000)):
            raise ValueError(
                "Inference steps should be an integer in range [1, 1000]. "
                f"Current value is {self.steps}"
            )
