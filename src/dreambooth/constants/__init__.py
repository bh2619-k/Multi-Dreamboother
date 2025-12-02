from dataclasses import dataclass


@dataclass(frozen=True)
class SupportedModels:
    hf_ids = [
        "stabilityai/stable-diffusion-2-1",
    ]  # ID of model on hugging-face


@dataclass(frozen=True)
class DefaultModelResolution:
    model_res = {
        "stabilityai/stable-diffusion-2-1": 768,
    }
