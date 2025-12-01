from typing import Callable

import torch
from PIL import Image
from torchvision import transforms as T

from dreambooth import constants


def get_transform_image_to_tensor(model_name: str, train_inference: str = "train"):
    """
    Return a torchvision transform pipeline depending on model and mode.

    Args:
        model_name (str): Name of the model (e.g. 'stabilityai/stable-diffusion-2-1')
        train_inference (str): 'train', 'val', or 'test'

    Returns:
        Callable: torchvision transform pipeline

    Raises:
        ValueError: If the model name is not supported.
    """
    # --- Determine image size based on known models ---
    if "stabilityai/stable-diffusion-2-1" in model_name:
        image_size = 768
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models: {constants.SupportedModels.hf_ids}"
        )

    # Define transforms
    if train_inference == "train":
        transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif train_inference in {"val", "test"}:
        transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    else:
        raise ValueError(f"Invalid mode '{train_inference}'. Expected 'train', 'val', or 'test'.")

    return transform


def get_transform_tensor_to_pil() -> Callable:
    """
    Returns a function that converts a normalized tensor image ([-1, 1] range)
    back to a PIL image ([0, 255] range).

    Works as the inverse of:
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.ToTensor()
    """

    def reverse_transform(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a normalized tensor (C x H x W) to a PIL image.

        Args:
            tensor (torch.Tensor): Image tensor, normalized to [-1, 1].

        Returns:
            PIL.Image: Converted image.
        """
        if tensor.ndim == 4:  # batch of images: take first
            tensor = tensor[0]

        # Unnormalize from [-1,1] to [0,1]
        tensor = (tensor * 0.5) + 0.5
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to PIL image
        to_pil = T.ToPILImage()
        return to_pil(tensor)

    return reverse_transform
