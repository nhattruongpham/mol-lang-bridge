import os
import torch
import random
import selfies as sf
from datasets import load_dataset
from rdkit import Chem
from torch.utils.data import DistributedSampler, DataLoader, Dataset


def get_dataloader(dataset, batchsize, rank, world_size):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    def collate(batch):
        tok_selfies = [i["tok_selfies"] for i in batch]
        caption_states = [i["caption_states"] for i in batch]
        caption_mask = [i["caption_mask"] for i in batch]
        corrupted_tok_selfies = [i["corrupted_tok_selfies"] for i in batch]
        return (
            torch.concat(tok_selfies, dim=0),
            torch.concat(caption_states, dim=0),
            torch.concat(caption_mask, dim=0),
            torch.concat(corrupted_tok_selfies, dim=0),
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
        replace_desc=False,
        pre=None,
        prob=0,
        load_state=True,
        corrupt_prob=0.4,
    ):
        super().__init__()
        self.dir = dir
        self.tokenizer = tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob = prob
        self.corrupt_prob = corrupt_prob
        self.ori_data = self.create_data()
        self.load_state = load_state
        self.desc_state = self.get_desc_state() if load_state else None

    def get_desc_state(self):
        file_path = os.path.join(self.dir, self.split + "_desc_states.pt")
        return torch.load(file_path)

    def create_data(self):
        dataset = load_dataset(
            "ndhieunguyen/LPM-24",
            token=True,
            split=self.split,
        )

        return [
            (int(sample_id), sample_selfies, sample_caption)
            for (sample_id, sample_selfies, sample_caption) in zip(
                dataset["id"], dataset["selfies"], dataset["caption"]
            )
        ]

    def __len__(self):
        return len(self.ori_data)

    def permute(self, smiles):
        if random.random() < self.prob:
            return changeorder(smiles, shuffle=True)
        else:
            return smiles

    def __getitem__(self, idx):
        data = self.ori_data[idx]
        sample = {"id": data[0], "selfies": self.permute(data[1]), "caption": data[2]}
        sample["tok_selfies"] = self.tokenizer(sample["selfies"])
        sample["corrupted_tok_selfies"] = sample["tok_selfies"]

        if self.load_state:
            sample["caption_states"] = self.desc_state[data[0]]["states"]
            sample["caption_mask"] = self.desc_state[data[0]]["mask"]

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
