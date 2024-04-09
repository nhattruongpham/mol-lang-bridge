import os
import torch
import argparse
from tqdm import tqdm
from transformers import T5EncoderModel

from src.scripts.mydatasets import Lang2molDataset
from src.scripts.mytokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True)
parser.add_argument("--data_dir", default="dataset")
parser.add_argument("--pretrained_name", default="QizhiPei/biot5-base-text2mol")
parser.add_argument("--hf_token", default="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")
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)
os.system(f"huggingface-cli login --token {args.hf_token}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = Tokenizer()
dataset = Lang2molDataset(
    dir=args.data_dir,
    tokenizer=tokenizer,
    split=args.split,
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
        last_hidden_state = model(input_ids=input_ids.to(device)).last_hidden_state
        volume[id] = {
            "states": last_hidden_state.to("cpu"),
            "mask": attention_mask,
        }

        if (i + 1) % 5000 == 0:
            torch.save(
                volume,
                os.path.join(
                    args.data_dir, args.split + f"_caption_states_{i-4999}_{i}.pt"
                ),
            )
            volume = {}
