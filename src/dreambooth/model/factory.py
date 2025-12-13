from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from dreambooth import constants


@dataclass
class SDComponents:
    tokenizer_1: Any
    text_encoder_1: Any
    unet: Any
    vae: Any
    noise_scheduler: Any

    # Names must match dataclass attributes
    trainables: list[str]
    frozen: list[str]
    cpu: list[str]

    _tok_grad_hook: Optional[Any] = None
    _special_token_id = None


    def _get_by_name(self, name: str) -> Any:
        """Return the component by its attribute name, or raise KeyError."""
        try:
            return getattr(self, name)
        except AttributeError as e:
            raise KeyError(f"Unknown component name: {name}") from e

    def _iter_modules(self, names: Iterable[str]) -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for known names; skip None components."""
        seen = set()
        for name_i in names:
            if name_i in seen:
                continue
            seen.add(name_i)
            m = self._get_by_name(name_i)
            if m is None:
                continue
            yield name_i, m

    def _freeze_by_config(self) -> SDComponents:
        """Set requires_grad according to (trainables, frozen). Doesn't change
        .train()/.eval()."""
        for _, m in self._iter_modules(self.trainables):
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = True

        for _, m in self._iter_modules(self.frozen):
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = False
        return self

    def _textual_inversion_text_encoder_1_freezing(self) -> SDComponents:
        """Set require_grad to false for all parameters of
        text encoder 1, except the token embedings"""
        te = self.text_encoder_1
        # Freeze all text encoder parameters
        te.requires_grad_(False)

        # Unfreeze full embedding matrix (needed for gradients)
        emb = te.get_input_embeddings()
        emb.weight.requires_grad = True

        return self

    def freeze_parameters(
        self, train_text_encoder: bool, text_encoder_training_mode: str
    ):

        if not train_text_encoder:
            return self._freeze_by_config()
        else:
            if text_encoder_training_mode == "full":
                # Freeze everything based on config
                self._freeze_by_config()
            elif text_encoder_training_mode == "textual_inversion":
                # Freeze everything based on config
                self._freeze_by_config()
                # For text encoder 1, freeze everything and unfreeze only token
                # embedding row
                self._textual_inversion_text_encoder_1_freezing()
            else:
                raise ValueError(
                    "text_encoder_training_mode is not valid, "
                    "should be 'full' or 'textual_inversion', "
                    f"current value: {text_encoder_training_mode}"
                )
            return self

    def text_encoder_1_token_gradient_masking(self):

        # Prevent duplicate hooks
        if self._tok_grad_hook is not None:
            raise RuntimeError(
                "Special token id is not set. Call expand_tokenizer_1() first."
            )

        # Prevent duplicate hooks
        if self._tok_grad_hook is not None:
            return
        special_id = self._special_token_id

        emb = self.text_encoder_1.get_input_embeddings()

        def hook_fn(grad):
            # grad: [vocab_size, hidden_dim]
            mask = torch.zeros_like(grad)
            mask[special_id] = grad[special_id]
            return mask

        # Save handle
        self._tok_grad_hook = emb.weight.register_hook(hook_fn)

    def set_train_mode(self) -> SDComponents:
        """Switch runtime mode only. No requires_grad changes; no device/dtype moves."""
        for _, m in self._iter_modules(self.trainables):
            if hasattr(m, "train"):
                m.train()
        for _, m in self._iter_modules(self.frozen):
            if hasattr(m, "eval"):
                m.eval()
        for _, m in self._iter_modules(self.cpu):
            if hasattr(m, "eval"):
                m.eval()
        return self

    def set_eval_mode(self) -> SDComponents:
        """Put all components into eval mode (useful for validation/inference)."""
        seen = set()
        for group in (self.trainables, self.frozen, self.cpu):
            for name, m in self._iter_modules(group):
                if name in seen:
                    continue
                seen.add(name)
                if hasattr(m, "eval"):
                    m.eval()
        return self

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield parameters that are marked requires_grad=True
        (after freeze_by_config)."""
        for _, m in self._iter_modules(self.trainables):
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

    def get_trainable_models(self) -> Iterator[tuple[str, Any]]:
        """Iterator of (name, module) pairs for trainable models."""
        yield from self._iter_modules(self.trainables)

    def get_frozen_models(self) -> Iterator[tuple[str, Any]]:
        """Iterator of (name, module) pairs for frozen models."""
        yield from self._iter_modules(self.frozen)

    def get_component_groups(self) -> dict[str, list[str]]:
        """Return a copy of the component grouping configuration."""
        return {
            "trainables": list(self.trainables),
            "frozen": list(self.frozen),
            "cpu": list(self.cpu),
        }

    def get_tokenizer_1(self):
        return self.tokenizer_1

    def get_text_encoder_1(self):
        return self.text_encoder_1

    def get_unet(self):
        return self.unet

    def get_vae(self):
        return self.vae

    def get_noise_scheduler(self):
        return self.noise_scheduler

    def expand_tokenizer_1(self, special_token: str):
        num_added = self.tokenizer_1.add_tokens(special_token)
        if num_added == 0:
            raise ValueError(
                f"Special token {special_token!r} already exists in tokenizer."
            )
        self.text_encoder_1.resize_token_embeddings(len(self.tokenizer_1))
        special_token_id = self.tokenizer_1.convert_tokens_to_ids(special_token)
        self._special_token_id = special_token_id

    def __post_init__(self):
        names = {"tokenizer_1", "text_encoder_1", "unet", "vae", "noise_scheduler"}
        for label, lst in (
            ("trainables", self.trainables),
            ("frozen", self.frozen),
            ("cpu", self.cpu),
        ):
            unknown = [n for n in lst if n not in names]
            if unknown:
                raise ValueError(f"{label} contains unknown names: {unknown}")
        s_train, s_frozen, s_cpu = map(set, (self.trainables, self.frozen, self.cpu))
        overlap = (s_train & s_frozen) | (s_train & s_cpu) | (s_frozen & s_cpu)
        if overlap:
            raise ValueError(
                f"Component lists must be disjoint; overlaps: {sorted(overlap)}"
            )

    def export_for_inference(
        self,
        out_dir: Path,
        metadata: dict,
        save_text_encoder: bool,
        accelerator: Optional[Accelerator],
    ):
        """
        Export diffution model weights into a Diffusers-compatible folder
        so they can be loaded directly with StableDiffusionPipeline.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap models from DDP/FSDP wrappers before saving
        if accelerator is None:
            unet = self.get_unet()
        else:
            unet = accelerator.unwrap_model(self.get_unet())
        unet.save_pretrained(out_dir / "unet")

        # Save tokenizer
        tokenizer = self.get_tokenizer_1()
        tokenizer.save_pretrained(out_dir / "tokenizer")

        if save_text_encoder:
            te = self.get_text_encoder_1()
            if accelerator is not None:
                te = accelerator.unwrap_model(te)
            te.save_pretrained(out_dir / "text_encoder")

        # Optional metadata
        torch.save(
            metadata,
            out_dir / "meta.pt",
        )


def _filter_args(source: dict, allowed: list[str]) -> dict:
    return {k: v for k, v in source.items() if k in set(allowed)}


def build_model_components(model_name: str, **kwargs) -> SDComponents:
    """Dispatch to the appropriate builder based on the base model repo id."""
    if "stable-diffusion-v1-5/stable-diffusion-v1-5" in model_name.lower():
        return _build_stable_diffusion_v1x_base_components(
            model_name=model_name, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models: {constants.SupportedModels.hf_ids}"
        )


def _build_stable_diffusion_v1x_base_components(
    model_name: str, **kwargs
) -> SDComponents:
    """Build Stable Diffusion 1.x model components directly (no pipeline)
    for DreamBooth training.

    Loads the tokenizer, text encoder, VAE, U-Net, and noise scheduler,
    constructs a SDComponents config object, freezes non-trainable modules,
    and sets training mode.

    Args:
        model_name:
            Hugging Face model repo ID or local path for a Stable Diffusion 2.1
            variant.
        **kwargs:
            Arbitrary keyword arguments forwarded to the component builder.
            Must include:
                - train_text_encoder (bool): Whether the CLIP text encoder should
                  be trainable.

    Returns:
        SDComponents:
            A configuration object containing loaded model components and
            training/freeze state.
    """

    # --- Required argument validation ---
    if "train_text_encoder" not in kwargs:
        raise KeyError("`train_text_encoder` is missing.")

    # --- Required arguments
    fargs = _filter_args(
        source=kwargs,
        allowed=["train_text_encoder", "text_encoder_training_mode", "special_token"],
    )
    special_token = fargs["special_token"]
    train_text_encoder: bool = fargs["train_text_encoder"]
    train_text_encoder_mode: bool = fargs["text_encoder_training_mode"]
    if train_text_encoder:
        if train_text_encoder_mode not in ["full", "textual_inversion"]:
            raise ValueError(
                "text encoder mode should be either 'full' or "
                f"'textual_inversion', got {train_text_encoder_mode}"
            )

    # --- Load components directly (no pipeline) ---
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
    )

    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    unet_dtype = torch.float32
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        torch_dtype=unet_dtype,
        variant=None,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    components = SDComponents(
        tokenizer_1=tokenizer,
        text_encoder_1=text_encoder,
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        trainables=["unet"] + (["text_encoder_1"] if train_text_encoder else []),
        frozen=["vae"] + (["text_encoder_1"] if not train_text_encoder else []),
        cpu=["noise_scheduler", "tokenizer_1"],
    )

    if train_text_encoder and train_text_encoder_mode == "textual_inversion":
        components.expand_tokenizer_1(special_token=special_token)

    components.freeze_parameters(
        train_text_encoder=train_text_encoder,
        text_encoder_training_mode=train_text_encoder_mode,
    )
    components.set_train_mode()

    return components
