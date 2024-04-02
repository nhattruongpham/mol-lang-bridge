import torch
import os
from os import path as osp
import regex
import random
from transformers import AutoTokenizer


################################
def getrandomnumber(numbers, k, weights=None):
    if k == 1:
        return random.choices(numbers, weights=weights, k=k)[0]
    else:
        return random.choices(numbers, weights=weights, k=k)


# simple smiles tokenizer
# treat every charater as token
def build_simple_smiles_vocab(dir):
    assert dir is not None, "dir and smiles_vocab can not be None at the same time."
    if not osp.exists(osp.join(dir, "simple_smiles_tokenizer_vocab.txt")):
        # print('Generating Vocabulary for {} ...'.format(dir))
        dirs = list(
            osp.join(dir, i) for i in ["train.txt", "validation.txt", "test.txt"]
        )
        smiles = []
        for idir in dirs:
            with open(idir, "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    line = line.split("\t")
                    assert len(line) == 3, "Dataset format error."
                    if line[1] != "*":
                        smiles.append(line[1].strip())
        char_set = set()
        for smi in smiles:
            for c in smi:
                char_set.add(c)
        vocabstring = "".join(char_set)
        with open(osp.join(dir, "simple_smiles_tokenizer_vocab.txt"), "w") as f:
            f.write(osp.join(vocabstring))
        return vocabstring
    else:
        print("Reading in Vocabulary...")
        with open(osp.join(dir, "simple_smiles_tokenizer_vocab.txt"), "r") as f:
            vocabstring = f.readline().strip()
        return vocabstring


def get_tokenizer(name="QizhiPei/biot5-base-text2mol"):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    amino_acids = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    prefixed_amino_acids = [f"<p>{aa}" for aa in amino_acids]
    tokenizer.add_tokens(prefixed_amino_acids)

    selfies_dict_list = [line.strip() for line in open("selfies_dict.txt")]
    tokenizer.add_tokens(selfies_dict_list)

    special_tokens_dict = {
        "additional_special_tokens": [
            "<bom>",
            "<eom>",
            "<bop>",
            "<eop>",
            "MOLECULE NAME",
            "DESCRIPTION",
            "PROTEIN NAME",
            "FUNCTION",
            "SUBCELLULAR LOCATION",
            "PROTEIN FAMILIES",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


if __name__ == "__main__":
    import selfies as sf

    tok = get_tokenizer()
    smiles = [
        "[210Po]",
        "C[C@H]1C(=O)[C@H]([C@H]([C@H](O1)OP(=O)(O)OP(=O)(O)OC[C@@H]2[C@H](C[C@@H](O2)N3C=C(C(=O)NC3=O)C)O)O)O",
        "C(O)P(=O)(O)[O-]",
        "CCCCCCCCCCCC(=O)OC(=O)CCCCCCCCCCC",
        "C[C@]12CC[C@H](C[C@H]1CC[C@@H]3[C@@H]2CC[C@]4([C@H]3CCC4=O)C)O[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)C(=O)O)O)O)O",
    ]
    selfies = [sf.encoder(smiles_ele) for smiles_ele in smiles]
    print(tok(selfies))
