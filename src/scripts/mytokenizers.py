import os
import torch
import random
import selfies as sf
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
    if not os.path.exists(os.path.join(dir, "simple_smiles_tokenizer_vocab.txt")):
        # print('Generating Vocabulary for {} ...'.format(dir))
        dirs = list(
            os.path.join(dir, i) for i in ["train.txt", "validation.txt", "test.txt"]
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
        with open(os.path.join(dir, "simple_smiles_tokenizer_vocab.txt"), "w") as f:
            f.write(os.path.join(vocabstring))
        return vocabstring
    else:
        print("Reading in Vocabulary...")
        with open(os.path.join(dir, "simple_smiles_tokenizer_vocab.txt"), "r") as f:
            vocabstring = f.readline().strip()
        return vocabstring


class Tokenizer:
    def __init__(
        self,
        pretrained_name="QizhiPei/biot5-base-text2mol",
        selfies_dict_path=os.path.join("dataset", "selfies_dict.txt"),
    ):
        self.tokenizer = self.get_tokenizer(pretrained_name, selfies_dict_path)

    def get_tokenizer(self, pretrained_name, selfies_dict_path):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
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

        selfies_dict_list = [line.strip() for line in open(selfies_dict_path)]
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

    def __call__(self, *args, **kwds):
        return self.tokenizer(*args, **kwds)

    def __len__(self):
        return len(self.tokenizer)

    def corrupt(self, selfies_list: list):
        tensors = []
        if type(selfies_list) is str:
            selfies_list = [selfies_list]
        for selfies in selfies_list:
            tensors.append(self.corrupt_one(selfies))
        return torch.concat(tensors, dim=0)

    # TODO: rewrite this for selfies
    def corrupt_one(self, selfies):
        smi = sf.decoder(selfies)
        # res = [self.toktoid[i] for i in self.rg.findall(smi)]
        res = [i for i in self.rg.findall(smi)]
        total_length = len(res) + 2
        if total_length > self.max_len:
            return self.encode_one(smi)
        ######################## start corruption ###########################
        r = random.random()
        if r < 0.3:
            pa, ring = True, True
        elif r < 0.65:
            pa, ring = True, False
        else:
            pa, ring = False, True
        #########################
        max_ring_num = 1
        ringpos = []
        papos = []
        for pos, at in enumerate(res):
            if at == "(" or at == ")":
                papos.append(pos)
            elif at.isnumeric():
                max_ring_num = max(max_ring_num, int(at))
                ringpos.append(pos)
        # ( & ) remove
        r = random.random()
        if r < 0.3:
            remove, padd = True, True
        elif r < 0.65:
            remove, padd = True, False
        else:
            remove, padd = False, True
        if pa and len(papos) > 0:
            if remove:
                # remove pa
                n_remove = getrandomnumber(
                    [1, 2, 3, 4], 1, weights=[0.6, 0.2, 0.1, 0.1]
                )
                p_remove = set(random.choices(papos, weights=None, k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p] = None
                    # print('debug pa delete {}'.format(p))
        # Ring remove
        r = random.random()
        if r < 0.3:
            remove, radd = True, True
        elif r < 0.65:
            remove, radd = True, False
        else:
            remove, radd = False, True
        if ring and len(ringpos) > 0:
            if remove:
                # remove ring
                n_remove = getrandomnumber(
                    [1, 2, 3, 4], 1, weights=[0.7, 0.2, 0.05, 0.05]
                )
                p_remove = set(random.choices(ringpos, weights=None, k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p] = None
                    # print('debug ring delete {}'.format(p))
        # ring add & ( ) add
        if pa:
            if padd:
                n_add = getrandomnumber([1, 2, 3], 1, weights=[0.8, 0.2, 0.1])
                n_add = min(self.max_len - total_length, n_add)
                for _ in range(n_add):
                    sele = random.randrange(len(res) + 1)
                    res.insert(sele, "(" if random.random() < 0.5 else ")")
                    # print('debug pa add {}'.format(sele))
                    total_length += 1
        if ring:
            if radd:
                n_add = getrandomnumber([1, 2, 3], 1, weights=[0.8, 0.2, 0.1])
                n_add = min(self.max_len - total_length, n_add)
                for _ in range(n_add):
                    sele = random.randrange(len(res) + 1)
                    res.insert(sele, str(random.randrange(1, max_ring_num + 1)))
                    # print('debug ring add {}'.format(sele))
                    total_length += 1

        ########################## end corruption ###############################
        # print('test:',res)
        # print('test:',''.join([i for i in res if i is not None]))

        res = [self.toktoid[i] for i in res if i is not None]
        res = [1] + res + [2]
        if len(res) < self.max_len:
            res += [0] * (self.max_len - len(res))
        else:
            res = res[: self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])

    def decode(self, sample):
        return self.tokenizer.decode(sample)

if __name__ == "__main__":
    import selfies as sf

    tokenizer = Tokenizer(
        selfies_dict_path=r"D:\molecule\mol-lang-bridge\dataset\selfies_dict.txt"
    )
    smiles = [
        "[210Po]",
        "C[C@H]1C(=O)[C@H]([C@H]([C@H](O1)OP(=O)(O)OP(=O)(O)OC[C@@H]2[C@H](C[C@@H](O2)N3C=C(C(=O)NC3=O)C)O)O)O",
        "C(O)P(=O)(O)[O-]",
        "CCCCCCCCCCCC(=O)OC(=O)CCCCCCCCCCC",
        "C[C@]12CC[C@H](C[C@H]1CC[C@@H]3[C@@H]2CC[C@]4([C@H]3CCC4=O)C)O[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)C(=O)O)O)O)O",
    ]
    selfies = [sf.encoder(smiles_ele) for smiles_ele in smiles]
    output = tokenizer(
        selfies,
        max_length=512,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    print(output["input_ids"])
