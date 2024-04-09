import os
import torch
import random
import selfies as sf
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import DistributedSampler, DataLoader, Dataset
import glob


def get_dataloader(dataset, batchsize, rank, world_size):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    def collate(batch):
        selfies_ids = [i["selfies_ids"] for i in batch]
        caption_state = [i["caption_state"] for i in batch]
        caption_mask = [i["caption_mask"] for i in batch]
        corrupted_selfies_ids = [i["corrupted_selfies_ids"] for i in batch]
        return (
            torch.concat(selfies_ids, dim=0),
            torch.concat(caption_state, dim=0),
            torch.concat(caption_mask, dim=0),
            torch.concat(corrupted_selfies_ids, dim=0),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler,
    )

    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                yield i
            ec += 1

    return iter(cycle())


class Lang2molDataset(Dataset):
    def __init__(
        self,
        dir,
        tokenizer,
        split,
        pre=None,
        prob=0,
        load_state=True,
        corrupt_prob=0.4,
    ):
        super().__init__()
        self.dir = dir
        self.tokenizer = tokenizer
        self.split = split
        self.pre = pre
        self.prob = prob
        self.corrupt_prob = corrupt_prob
        self.ori_data = self.create_data()
        self.load_state = load_state
        self.caption_state = self.get_caption_state() if load_state else None

    def get_caption_state(self):
        all_caption_state = []
        file_paths = glob.glob(os.path.join(self.dir, "*_caption_states.pt"))
        for file_path in file_paths:
            all_caption_state.append(torch.load(file_path))
        return torch.cat(all_caption_state, dim=0)

    def create_data(self):
        try:
            dataset = load_dataset(
                "ndhieunguyen/LPM-24",
                token=True,
                split=self.split,
            ).sort("id")
        except:
            dataset = load_dataset(
                "ndhieunguyen/LPM-24",
                use_auth_token=True,
                split=self.split,
            ).sort("id")

        return [
            (int(sample_id), sample_selfies, sample_caption)
            for (sample_id, sample_selfies, sample_caption) in zip(
                dataset["id"], dataset["selfies"], dataset["caption"]
            )
        ]

    def __len__(self):
        return len(self.ori_data)

    def permute(self, selfies):
        if random.random() < self.prob:
            return changeorder(selfies, shuffle=True)
        else:
            return selfies

    def __getitem__(self, idx):
        data = self.ori_data[idx]
        sample = {"id": data[0], "selfies": self.permute(data[1]), "caption": data[2]}
        sample["selfies_ids"] = self.tokenizer(
            sample["selfies"],
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).input_ids
        sample["corrupted_selfies_ids"] = sample["selfies_ids"]
        # TODO: uncomment this
        # sample["corrupted_selfies_ids"] = (
        #     self.tokenizer.corrupt(sample["selfies"])
        #     if random.random() < self.corrupt_prob
        #     else sample["selfies_ids"]
        # )

        if self.load_state:
            sample["caption_state"] = self.caption_state[data[0]]["states"]
            sample["caption_mask"] = self.caption_state[data[0]]["mask"]

        return sample


def changeorder(selfies, shuffle):
    smiles = sf.encoder(selfies)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return selfies
    Chem.Kekulize(mol)
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    if shuffle:
        random.shuffle(atom_indices)
    reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
    new_smiles = Chem.MolToSmiles(reordered_mol, kekuleSmiles=True)
    new_selfies = sf.decoder(new_smiles)

    return new_selfies


if __name__ == "__main__":
    from mytokenizers import Tokenizer

    tokenizer = Tokenizer(
        selfies_dict_path=r"D:\molecule\mol-lang-bridge\dataset\selfies_dict.txt"
    )
    train_dataset = Lang2molDataset(
        dir="dataset",
        tokenizer=tokenizer,
        split="train",
        load_state=False,
    )

    print(next(iter(train_dataset)))
