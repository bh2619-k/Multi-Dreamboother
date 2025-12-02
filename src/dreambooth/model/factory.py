from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
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

    def freeze_by_config(self) -> SDComponents:
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


def _filter_args(source: dict, allowed: list[str]) -> dict:
    return {k: v for k, v in source.items() if k in set(allowed)}


def build_model_components(model_name: str, **kwargs) -> SDComponents:
    """Dispatch to the appropriate builder based on the base model repo id."""
    if "stabilityai/stable-diffusion-2-1" in model_name.lower():
        return _build_stable_diffusion_v2x_base_components(
            model_name=model_name, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models: {constants.SupportedModels.hf_ids}"
        )


def _build_stable_diffusion_v2x_base_components(
    model_name: str, **kwargs
) -> SDComponents:
    """Build Stable Diffusion 1.x / 2.x model components directly (no pipeline)
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
    fargs = _filter_args(source=kwargs, allowed=["train_text_encoder"])
    train_text_encoder: bool = fargs["train_text_encoder"]

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

    components.freeze_by_config()
    components.set_train_mode()

    return components
