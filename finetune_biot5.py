from datasets import load_dataset
import selfies as sf
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
import wandb
from datetime import datetime
from zoneinfo import ZoneInfo
from accelerate import Accelerator
from argparse import ArgumentParser
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import glob

from src.dataset import MyDataset
from src.steps import train, validate


def main(args):
    # login wandb
    wandb.login(key=args.wandb_key)
    wandb.init(
        project="BioT5 finetuning",
        group=f"{datetime.now(tz=ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y/%m/%d_%H-%M-%S')}",
    )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_name, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_name)
    try:
        pretrained_weights = sorted(glob.glob("weights/*.pt"))[-1]
        model.load_state_dict(torch.load(pretrained_weights))
    except:
        pass
    model.train()
    model.to(device)
    print(f"Finish loading model from {args.pretrained_name}")

    # Prepare dataset
    dataset = load_dataset("ndhieunguyen/LPM-24", use_auth_token=True)
    train_data = dataset["train"]
    val_data = dataset["validation"]
    train_dataset = MyDataset(
        train_data, tokenizer, args.caption_max_length, args.selfies_max_length
    )
    val_dataset = MyDataset(
        val_data, tokenizer, args.caption_max_length, args.selfies_max_length
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = train(model, train_dataloader, tokenizer, optimizer)
        train_result = validate(model, train_dataloader, tokenizer, device)
        val_result = validate(model, val_dataloader, tokenizer, device, prefix="val")

        train_dict = {}
        train_dict["train_loss"] = total_loss
        train_dict.update(train_result)
        train_dict.update(val_result)

        wandb.log(train_dict)

        torch.save(model.state_dict(), f"weights/epoch_{str(epoch).rjust(2, '0')}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=65536)
    parser.add_argument("--final_cosine", type=int, default=1e-5)
    parser.add_argument("--lr", type=int, default=2e-2)
    parser.add_argument("--caption_max_length", type=int, default=512)
    parser.add_argument("--selfies_max_length", type=int, default=512)
    parser.add_argument(
        "--pretrained_name",
        type=str,
        default="QizhiPei/biot5-base-text2mol",
    )

    args = parser.parse_args()
    main(args)
