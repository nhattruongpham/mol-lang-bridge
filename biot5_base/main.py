import torch
from transformers import AutoTokenizer
from dataset import get_dataloaders
import lightning as pl
from lightning_module import BioT5Model
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import ArgumentParser


def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    train_dataloader = get_dataloaders(tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='train')
    val_dataloader = get_dataloaders(tokenizer, batch_size=1, num_workers=4, split='validation')

    model = BioT5Model(args, tokenizer, len(train_dataloader))
    model.to(device)

    ckpt_callback = ModelCheckpoint(
        dirpath='ckpt/',
        filename='ckpt_{eval_loss}',
        save_top_k=3,
        verbose=True,
        monitor='eval_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        callbacks=[ckpt_callback],
        max_epochs=args.epochs,
        accelerator='cuda' if args.cuda else 'cpu',
        devices=args.num_devices,
        precision='16', # 32 if has more vram
        gradient_clip_val=10.0
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
   
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='QizhiPei/biot5-base')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=int, default=0.1)
    
    args = parser.parse_args()
    
    main(args)