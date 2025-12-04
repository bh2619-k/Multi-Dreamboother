from dataclasses import dataclass


@dataclass(frozen=True)
class SupportedModels:
    hf_ids = [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ]  # ID of model on hugging-face


@dataclass(frozen=True)
class DefaultModelResolution:
    model_res = {
        "stable-diffusion-v1-5/stable-diffusion-v1-5": 512,
    }
