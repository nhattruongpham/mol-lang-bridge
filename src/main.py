import torch
from transformers import AutoTokenizer
from dataset_module import get_dataloaders
import lightning as pl
from lightning_module import T5MultimodalModel
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser, Namespace
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import yaml
import os
from utils import set_nested_attr

def main(args):
    seed_everything(42)
    device = torch.device('cuda' if args.cuda else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
    tokenizer.add_tokens(['α', 'β', 'γ', '<boc>', '<eoc>']) # Add greek symbol, <boc> is start_of_caption, <eoc> is end_of_caption
    train_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='train')
    val_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='validation')
    
    args.train_data_len = len(train_dataloader) // args.grad_accum
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    model = T5MultimodalModel(args)
    model.resize_token_embeddings(len(tokenizer)) ## Resize due to adding new tokens
    model.to(device)
    
    print(model)

    ckpt_callback = ModelCheckpoint(
        dirpath=args.output_folder,
        filename='ckpt_{eval_loss}',
        save_top_k=5,
        verbose=True,
        monitor='eval_loss',
        mode='min'
    )
    
    wandb_logger = WandbLogger(log_model=False,
                               project='ACL_Mol2Lang',
                               name=os.path.splitext(os.path.basename(args.model_config))[0]
                            )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        callbacks=[ckpt_callback, lr_monitor],
        max_epochs=args.epochs,
        accelerator='cuda' if args.cuda else 'cpu',
        devices=args.num_devices,
        precision=args.precision, # 32 if has more vram
        gradient_clip_val=10.0,
        logger=[wandb_logger],
        accumulate_grad_batches=args.grad_accum,
        deterministic=True
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
   
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--dataset_name_or_path', type=str, default='duongttr/chebi-20-new')
    parser.add_argument('--model_config', type=str, default='src/configs/config_use_v_nofg.yaml')
    parser.add_argument('--output_folder', type=str, default='ckpt/')
    
    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)
        
    main(args)