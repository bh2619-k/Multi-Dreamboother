from typing import Callable, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from dreambooth.data.data_utils import list_files, read_image_rgb


class ImageCollectionDatasetWithPrompts(Dataset):
    """Loads all RGB images from a folder and assigns a 
        prompt to each one according to inputed prompts.

    Each sample returns tokenized prompt IDs and image tensors 
    for DreamBooth-style training.
    """

    def __init__(
        self,
        folder_path: str,
        prompts: Union[str, list[str]],
        tokenizer: PreTrainedTokenizer,
        transform: Optional[Callable] = None,
    ):

        self.folder_path = folder_path
        self.transform = transform
        self.prompts = prompts
        self.tokenizer = tokenizer

        self.image_files = list_files(
            folder=self.folder_path, suffixes=[".jpg", ".jpeg", ".png"]
        )

        # Expand prompts of it is only one string
        if isinstance(self.prompts, str):
            self.prompts = [self.prompts] * len(self.image_files)
        elif isinstance(self.prompts, list) and len(self.prompts) == 1:
            self.prompts = self.prompts * len(self.image_files)
        elif not isinstance(self.prompts, list):
            raise ValueError('input prompts should be a string or'
            ' a list of strings.')
        if len(self.image_files) != len(self.prompts):
            raise ValueError(
                "Number of images in the folder should be equal "
                f"number of prompts.\nFolder: {self.folder_path}"
                f"\nNumber of prompts: {len(self.prompts)}"
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
        prompt = self.prompts[index]
        sample['instance_prompt_ids'] = self.tokenizer(
            prompt,
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
