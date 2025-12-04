import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import tqdm as accel_tqdm
from torch.utils.data import DataLoader

from dreambooth import constants
from dreambooth.data.dataloader import get_image_collection_const_prompt_dataloader
from dreambooth.logger.logger_factory import LoggerFactory
from dreambooth.model.factory import SDComponents, build_model_components

log = LoggerFactory.get_logger()


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
        num_worker: int,
        batch_size: int,
        max_steps: int,
        learning_rate: float,
        prior_preservation: bool,
        num_class_images: int,
        class_generation_batch_size: int,
        seed: int,
    ):
        self.model_name = model_name
        self.input_folder = Path(input_folder)
        self.save_folder = Path(save_folder)
        self.train_text_encoder = train_text_encoder
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.prior_preservation = prior_preservation
        self.num_class_images = num_class_images
        self.class_generation_batch_size = class_generation_batch_size
        self.seed = int(seed)

        self._validate_input_arguments()

        # HF Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Set seed for reproducebility
        self._setup_seed(self.seed)

        # Global optimization step counter
        self.global_step: int = 0

        self.components: SDComponents | None = None
        self.build_components()

        self.train_data_loader: DataLoader | None = None
        self.create_train_data_loader()

        self.optimizer: torch.optim.Optimizer | None = None

        # Whether objects wraped by accelator
        self._acc_wrap = False

    # --------------------------- Data -----------------------------------

    def create_train_data_loader(self):
        """Construct the training dataloader from a folder of subject images."""
        components = self.get_components(check_wrap_with_accelarate=False)
        self.train_data_loader = get_image_collection_const_prompt_dataloader(
            folder_path=self.input_folder,
            num_workers=self.num_worker,
            model_name=self.model_name,
            batch_size=self.batch_size,
            mode="train",
            instance_prompt=self.instance_prompt,
            tokenizer_1=components.get_tokenizer_1(),
        )
        log.info(f"Number of batches in dataloader: {len(self.train_data_loader)}")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_data_loader is None:
            raise RuntimeError("Train dataloader is not initialized.")
        return self.train_data_loader

    # --------------------------- Models -----------------------------------

    def build_components(self) -> None:
        """Create/restore the SD components bundle."""
        self.components = build_model_components(
            model_name=self.model_name,
            train_text_encoder=self.train_text_encoder,
        )
        log.info(
            f"Model components are loaded: {self.components.get_component_groups()}"
        )

    def get_components(self, check_wrap_with_accelarate: bool = True) -> SDComponents:
        if check_wrap_with_accelarate:
            self._require_wrapping_with_accelarate()
        if self.components is None:
            raise RuntimeError("Models are not initialized.")
        return self.components

    def configure_optimizer(self):
        """Configure the AdamW optimizer with sane defaults and config overrides."""
        components = self.get_components(check_wrap_with_accelarate=False)
        params = list(components.trainable_parameters())
        if not params:
            raise RuntimeError("No trainable parameters found.")
        wd = 0.01
        betas = (0.9, 0.999)
        eps = 1e-8
        lr = self.learning_rate
        self.optimizer = torch.optim.AdamW(
            [{"params": params, "lr": lr, "weight_decay": wd}], betas=betas, eps=eps
        )

    def get_optimizer(
        self, check_wrap_with_accelarate: bool = True
    ) -> torch.optim.Optimizer:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")
        return self.optimizer

    def prepare_with_accelerate(self):
        """Wrap trainable and eval-only modules with Accelerate once."""
        comps = self.get_components(check_wrap_with_accelarate=False)

        # Collect trainable modules by (name, module)
        train_te = self.train_text_encoder
        trainable_pairs = [("unet", comps.unet)]
        if train_te:
            trainable_pairs.append(("text_encoder_1", comps.text_encoder_1))

        # Build the argument list for prepare()
        prepare_args = [m for _, m in trainable_pairs] + [
            self.optimizer,
            self.train_data_loader,
        ]

        # Run prepare() once
        prepared = self.accelerator.prepare(*prepare_args)

        # Unpack back: modules (same order), then optimizer, then dataloader
        *prepared_modules, self.optimizer, self.train_data_loader = prepared
        for (name, _), wrapped in zip(trainable_pairs, prepared_modules, strict=False):
            setattr(comps, name, wrapped)

        # Eval-only modules via prepare_model (loop, no branches)
        eval_only_names = ["vae"] + ([] if train_te else ["text_encoder_1"])
        for name in eval_only_names:
            m = getattr(comps, name)
            setattr(
                comps,
                name,
                self.accelerator.prepare_model(m, evaluation_mode=True),
            )

        self._acc_wrap = True

    def prepare(self, *objs):
        return self.accelerator.prepare(*objs)

    def _require_wrapping_with_accelarate(self) -> None:
        if not self._acc_wrap:
            raise RuntimeError("Module is not wrapped with accelarate.")

    # --------------------------- Training ---------------------------------

    def compute_loss(
        self, gt_noise: torch.Tensor, pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """Standard MSE noise-prediction loss (epsilon-prediction)."""
        return F.mse_loss(pred_noise, gt_noise, reduction="none").mean([1, 2, 3]).mean()

    def train_step(self, batch):
        """Run one forward/backward update."""
        comps = self.get_components(check_wrap_with_accelarate=True)
        optim = self.get_optimizer(check_wrap_with_accelarate=True)
        accumulate_models = [m for _, m in comps.get_trainable_models()]
        if not accumulate_models:
            raise RuntimeError("No trainable models found.")

        train_te = self.train_text_encoder
        te_ctx = nullcontext() if train_te else torch.no_grad()

        images = batch["pixel_values"]
        tokens_ids = batch["input_ids"]

        with self.accelerator.accumulate(tuple(accumulate_models)):
            # Calculate latent of image with encoder of VAE
            with torch.no_grad():
                latents = comps.vae.encode(images).latent_dist.sample()
                # adjusts the latent magnitude
                latents = latents * comps.vae.config.scaling_factor

            # Sample noise that we add to the latents
            noise = torch.randn_like(latents)
            batch_size = noise.shape[0]

            # Sample random timestep for each image
            timesteps = torch.randint(
                low=0,
                high=comps.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=self.device,
            ).long()

            # Add Noise to latents, Diffusion forward pass
            noisy_latents = comps.noise_scheduler.add_noise(latents, noise, timesteps)

            # Embedding of text prompt
            with te_ctx:
                encoder_hidden_states = comps.text_encoder_1(tokens_ids)[0]

            # Predict the noise that was originally added to the clean image latent
            noise_pred = comps.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Calculate loss
            loss = self.compute_loss(gt_noise=noise, pred_noise=noise_pred)

            # Calculate gradients
            self.accelerator.backward(loss)

            # Clip gradients (distributed safe)
            for _, m in comps.get_trainable_models():
                self.accelerator.clip_grad_norm_(m.parameters(), max_norm=1.0)

            # Update weights
            optim.step()
            optim.zero_grad(set_to_none=True)
            self.inc_global_step()

            return loss.detach().item()

    def train(self) -> None:
        """Top-level training loop."""

        progress_bar = accel_tqdm(range(self.max_steps), desc="Training", leave=True)

        components = self.get_components(check_wrap_with_accelarate=False)

        # Freeze models that are not target for training
        components.freeze_by_config()

        self.configure_optimizer()

        # Wrap models with accelarate for disributed training
        self.prepare_with_accelerate()

        # Get newly wrapped with accelarate model compoentns
        components = self.get_components(check_wrap_with_accelarate=True)

        # Get newly wrapped with accelarate optimizer
        optim = self.get_optimizer()

        # Set non-freezed model in training mode and
        # other on eval mode
        components.set_train_mode()

        self.reset_global_step(init_value=0)
        optim.zero_grad(set_to_none=True)

        train_data_loader = self.get_train_dataloader()

        losses = []

        # Training loop
        # Train until reach max number of steps
        done = False
        while not done:
            for batch_i in train_data_loader:

                # Forward and backward pass with one mini batch
                loss = self.train_step(batch=batch_i)
                losses.append(loss)

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss_last": float(losses[-1]),
                        "loss_win_10": float(torch.mean(torch.tensor(losses[-10:]))),
                        "loss_win_50": float(torch.mean(torch.tensor(losses[-50:]))),
                        "loss_win_all": float(torch.mean(torch.tensor(losses))),
                    }
                )

                if self.get_global_step() >= self.max_steps:
                    done = True
                    break

        log.info("Training was finished.")
        progress_bar.close()

        self.export_for_inference(out_dir=Path(self.save_folder))
        self._training_active = False

    # --------------------------- Export & Inference ------------------------

    def export_for_inference(self, out_dir: Path):
        """
        Export DreamBooth fine-tuned weights into a Diffusers-compatible folder
        so they can be loaded directly with StableDiffusionPipeline.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get model compoentns
        comps = self.get_components()

        # Unwrap models from DDP/FSDP wrappers before saving
        unet = self.accelerator.unwrap_model(comps.unet)
        unet.save_pretrained(out_dir / "unet")

        if self.train_text_encoder:
            te = comps.text_encoder_1
            te = self.accelerator.unwrap_model(te) if hasattr(te, "module") else te
            te.save_pretrained(out_dir / "text_encoder")

        # Optional metadata
        model_res = constants.DefaultModelResolution.model_res[self.model_name]
        torch.save(
            {
                "training_method": "dreambooth",
                "model_name": self.model_name,
                "train_text_encoder": self.train_text_encoder,
                "global_step": int(self.global_step),
                "instance_prompt": self.instance_prompt,
                "prior_preservation": self.prior_preservation,
                "resolution": (model_res, model_res),
            },
            out_dir / "meta.pt",
        )

        log.info(f"Exporting models to {out_dir.resolve()} is done.")

    # --------------------------- Other -----------------------------------

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
        if not (isinstance(self.num_worker, int) and -1 <= self.num_worker <= 8):
            raise ValueError(
                "num_worker should be an integer in range [-1, 8]."
                f" Current value is {self.num_worker}"
            )
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

    def inc_global_step(self) -> None:
        self.global_step += 1

    def get_global_step(self) -> int:
        return self.global_step

    def reset_global_step(self, init_value=0) -> int:
        self.global_step = init_value
        return self.global_step

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
