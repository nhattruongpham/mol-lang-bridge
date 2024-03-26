import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(
        self, dataset, tokenizer, caption_max_length=512, selfies_max_length=512
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.caption_max_length = caption_max_length
        self.selfies_max_length = selfies_max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        caption, selfies = sample["caption"], sample["selfies"]
        caption_tokens = self.tokenizer(
            caption,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.caption_max_length,
            return_attention_mask=True,
        )
        selfies_tokens = self.tokenizer(
            selfies,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.selfies_max_length,
            return_attention_mask=True,
        )

        caption_ids = caption_tokens["input_ids"].squeeze()
        caption_mask = caption_tokens["attention_mask"].squeeze()
        selfies_ids = selfies_tokens["input_ids"].squeeze()

        return {
            "caption_ids": caption_ids.to(dtype=torch.long),
            "caption_mask": caption_mask.to(dtype=torch.long),
            "selfies_ids": selfies_ids.to(dtype=torch.long),
        }
