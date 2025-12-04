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
)
from PIL import Image

from dreambooth import constants
from dreambooth.data.data_utils import save_images
from dreambooth.logger.logger_factory import LoggerFactory

log = LoggerFactory.get_logger()


class PretrainedModelInference:
    """Common utilities and lifecycle for inference."""

    def __init__(
        self,
        model_name: str,
        prompts: list[str],
        batch_size: int,
        guidance_scale: float,
        steps: int,
        seed: int,
        save_folder: str,
    ):

        self.model_name = model_name
        self.prompts = prompts
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.save_folder = Path(save_folder)

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
        # Build pipeline with overrides so Accelerate can shard them
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
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
        pipe = self.get_pipe()

        all_prompts = self.prompts
        steps = self.steps
        guidance = self.guidance_scale

        model_res = constants.DefaultModelResolution.model_res[self.model_name]
        height, width = model_res, model_res

        batch_size = self.batch_size
        total_prompts = len(all_prompts)

        number_of_batches = math.ceil(total_prompts / batch_size)

        for batch_i in range(number_of_batches):
            with torch.inference_mode():

                start = batch_i * batch_size
                end = min(start + batch_size, total_prompts)

                batch_prompts = all_prompts[start:end]

                out = pipe(
                    prompt=batch_prompts,
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

    def generate_and_save_images(self) -> None:
        """
        Generate images in batches and save them to disk,
        logging progress every 20 images.
        """
        total_generated = 0

        for images in self.generate_image_batches():
            saved_paths = save_images(
                images, self.save_folder, ext="png", overwrite=True
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

        if self.model_name not in constants.SupportedModels.hf_ids:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                f"Supported models: {constants.SupportedModels.hf_ids}"
            )
        if not isinstance(self.prompts, list):
            raise ValueError(
                "Input prompts should be a list. " f"Current value: {self.prompts}"
            )
        if len(self.prompts) == 0:
            raise ValueError(
                "Input prompts should have at least one element. "
                f"Current value: {self.prompts}"
                f"Current length: {len(self.prompts)}"
            )
        for str_i in self.prompts:
            if not isinstance(str_i, str) or len(str_i) == 0:
                raise ValueError(
                    "Input prompts should be a non-empty valid strings. "
                    f"Current value: {self.prompts}, "
                    f"{str_i} violates the condition."
                )
        if self.batch_size < 1:
            raise ValueError(
                "Training batch size can not be an integer less than 1. "
                f"Current batch_size = {self.batch_size}."
            )
        if not isinstance(self.guidance_scale, (int, float)) or not (
            0 <= self.guidance_scale <= 10
        ):
            raise ValueError(
                "Guidance scale should be a float number between 0 and 10. "
                f"Current value is {self.guidance_scale}"
            )
        if not isinstance(self.steps, int) and not (1 <= self.steps <= 1000):
            raise ValueError(
                "Inference steps should be an integer in range [1, 1000]. "
                f"Current value is {self.steps}"
            )
        if not self.save_folder.is_dir():
            raise NotADirectoryError(f"Save folder invalid: {self.save_folder}")
