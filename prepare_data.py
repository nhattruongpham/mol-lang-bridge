import os
import torch
import argparse
from tqdm import tqdm
from transformers import T5EncoderModel

from src.scripts.mydatasets import Lang2molDataset
from src.scripts.mytokenizers import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="dataset")
parser.add_argument("--split", required=True)
parser.add_argument("--pretrained_name", default="QizhiPei/biot5-base-text2mol")
parser.add_argument("--hf_token", default="")
args = parser.parse_args()

os.system(f"huggingface-cli login --token {args.hf_token}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = get_tokenizer()
dataset = Lang2molDataset(
    dir=args.data_dir,
    tokenizer=tokenizer,
    split=args.split,
    replace_desc=False,
    load_state=False,
)
model = T5EncoderModel.from_pretrained(args.pretrained_name)
model.to(device)
model.eval()

volume = {}
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        id = dataset[i]["id"]
        caption = dataset[i]["caption"]
        output = tokenizer(
            caption,
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        last_hidden_state = model(input_ids=input_ids).last_hidden_state
        volume[id] = {
            "states": last_hidden_state.to("cpu"),
            "mask": attention_mask,
        }

torch.save(volume, os.path.join("dataset", args.split + "_desc_states.pt"))
