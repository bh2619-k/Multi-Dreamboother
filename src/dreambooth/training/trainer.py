from pathlib import Path

from dreambooth import constants


class DreamBoothTrainer:
    """Stable diffusion personalization with DreamBooth"""

    def __init__(
        self,
        model_name: str,
        input_folder: str,
        save_folder: str,
        train_text_encoder: bool,
        instance_prompt: str,
        class_prompt: str,
        batch_size: int,
        max_steps: int,
        learning_rate: float,
        prior_preservation: bool,
        num_class_images: int,
        class_generation_batch_size: int,
    ):
        self.model_name = model_name
        self.input_folder = Path(input_folder)
        self.save_folder = Path(save_folder)
        self.train_text_encoder = train_text_encoder
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.prior_preservation = prior_preservation
        self.num_class_images = num_class_images
        self.class_generation_batch_size = class_generation_batch_size

        self._validate_input_arguments()

    def _validate_input_arguments(
        self,
    ):
        if self.model_name not in constants.SupportedModels.hf_ids:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                f"Supported models: {constants.SupportedModels.hf_ids}"
            )
        if not self.input_folder.is_dir():
            raise NotADirectoryError(f"Input folder invalid: {self.input_folder}")
        if not self.save_folder.is_dir():
            raise NotADirectoryError(f"Save folder invalid: {self.save_folder}")
        if not isinstance(self.train_text_encoder, bool):
            raise ValueError("train_text_encoder should be a boolean.")
        if self.instance_prompt == "":
            raise ValueError("instance_prompt can not be empty sentence.")
        if self.class_prompt == "":
            raise ValueError("class_prompt can not be empty sentence.")
        if self.batch_size < 1:
            raise ValueError(
                "Training batch size can not be an integer less than 1. "
                f"Current batch_size = {self.batch_size}."
            )
        if self.prior_preservation and self.batch_size % 2 != 0:
            raise ValueError(
                f"Batch size must be EVEN when prior preservation is enabled. "
                f"Got batch_size={self.batch_size}."
            )
        if self.max_steps < 1:
            raise ValueError(
                "Number of steps should be greater equal to "
                f"one, current inputed value: {self.max_steps}"
            )
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError(
                "Learning rate should be a float number between "
                f"zero and one, current inputed value {self.learning_rate}."
            )
        if not isinstance(self.prior_preservation, bool):
            raise ValueError(
                "Prior preservation flag should be a boolean, "
                f"current value is {self.prior_preservation}."
            )
        if self.num_class_images < 1:
            raise ValueError(
                "Number of generated image with frozen stable "
                "diffusion should be greater equal to one."
                f" Current inputed value: {self.num_class_images}"
            )
        if self.class_generation_batch_size < 1:
            raise ValueError(
                "Batch size for generating images with frozen stable diffusion"
                " can not be an integer less than 1. "
                f"Current batch_size = {self.class_generation_batch_size}."
            )
