import os
from typing import Optional, Union

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from dreambooth.data.dataset import ImageCollectionDatasetWithPrompts
from dreambooth.data.transforms import get_transform_image_to_tensor


def get_image_collection_const_prompt_dataloader(
    folder_path: str,
    model_name: str,
    prompts: Union[str, list[str]],
    tokenizer_1: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 4,
    mode: str = "train",
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for images in a directory.

    Args:
        folder_path (str): Path to the folder containing image files.
        model_name (str): Name or identifier of the model; determines image
            size and transforms.
        prompts (Union[str, list[str]]): Text prompts associated with the 
            images. This may be a single string, in which case the same
            prompt is applied to all subject images, or a list of prompts
            whose length matches the number of subject images, assigning
            one prompt per image.
        tokenizer_1 (PreTrainedTokenizer): Tokenizer used to encode the 
            constant prompt.
        batch_size (int, optional): Number of images per batch. Defaults to 4.
        num_workers (int, optional): Number of subprocesses used for data 
            loading. Defaults to 4.
        mode (str, optional): One of {"train", "val", "test"}; affects shuffling
            and drop_last. Defaults to "train".
        shuffle (bool, optional): Whether to shuffle data. Defaults to True in 
            training mode, False otherwise.
        pin_memory (bool, optional): Whether to pin memory for faster GPU transfer. 
            Defaults to False.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """

    # Default shuffle logic
    if shuffle is None:
        shuffle = mode.lower() == "train"

    # Get appropriate transform
    transform = get_transform_image_to_tensor(
        model_name=model_name, 
        train_inference=mode
    )

    # Validate path
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Data folder '{folder_path}' does not exist.")

    # Initialize dataset
    dataset = ImageCollectionDatasetWithPrompts(
        folder_path=folder_path,
        transform=transform,
        prompts=prompts,
        tokenizer=tokenizer_1,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode.lower() == "train"),
        persistent_workers=(num_workers > 0 and mode.lower() == "train"),
        collate_fn=ImageCollectionDatasetWithPrompts.make_collate_fn(tokenizer_1),
    )

    return dataloader
