from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from dreambooth.data.data_utils import list_files, read_image_rgb


class ImageCollectionDatasetWithDefaultPrompt(Dataset):
    """Loads all RGB images from a folder and assigns a shared
      default prompt.

    Each sample returns tokenized prompt IDs and image tensors 
    for DreamBooth-style training.
    """

    def __init__(
        self,
        folder_path: str,
        instance_prompt: str,
        tokenizer: PreTrainedTokenizer,
        transform: Optional[Callable] = None,
    ):

        self.folder_path = folder_path
        self.transform = transform
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer

        self.image_files = list_files(
            folder=self.folder_path, suffixes=[".jpg", ".jpeg", ".png"]
        )

    def __len__(
        self,
    ) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):

        sample = {}

        # --- Image
        image = read_image_rgb(path=self.image_files[index])
        if self.transform is None:
            sample['image'] = image
        else:
            sample['image'] = self.transform(image)

        # --- Prompt
        sample['instance_prompt_ids'] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return sample

    @staticmethod
    def make_collate_fn(tokenizer: PreTrainedTokenizer):

        def collate_fn(samples):
            pixel_values = torch.stack([sample_i['image'] for sample_i in samples])
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()

            input_ids = [sample_i["instance_prompt_ids"] for sample_i in samples]
            input_ids = tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).input_ids

            batch = {
                'input_ids': input_ids,
                'pixel_values': pixel_values,
            }
            return batch

        return collate_fn
