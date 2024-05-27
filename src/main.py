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
    
    if args.dataset_name == 'lpm-24':
        args.dataset_name_or_path = 'ndhieunguyen/LPM-24'
    elif args.dataset_name == 'chebi-20':
        args.dataset_name_or_path = 'duongttr/chebi-20-new'
    else:
        raise Exception('Dataset name is invalid, please choose one in two: lpm-24, chebi-20')
    
    train_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='train')
    val_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='validation')
    
    if args.dataset_name == 'lpm-24':
        test_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='validation')
    elif args.dataset_name == 'chebi-20':
        test_dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split='test')
    
    args.train_data_len = len(train_dataloader) // args.grad_accum
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    model = T5MultimodalModel(args)
    model.resize_token_embeddings(len(tokenizer)) ## Resize due to adding new tokens
    model.to(device)
    model.tokenizer = tokenizer

    on_best_eval_loss_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_folder, "best_eval"),
        filename='ckpt_eval_loss_{eval_loss/dataloader_idx_0}',
        save_top_k=3,
        verbose=True,
        monitor='eval_loss/dataloader_idx_0',
        mode='min'
    )
    on_best_bleu2_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_folder, "best_bleu2"),
        filename='ckpt_bleu2_{bleu2/dataloader_idx_1}',
        save_top_k=3,
        verbose=True,
        monitor='bleu2/dataloader_idx_1',
        mode='max'
    )
    
    wandb_logger = WandbLogger(log_model=False,
                               project='ACL_Mol2Lang',
                               name=os.path.splitext(os.path.basename(args.model_config))[0]
                            )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        callbacks=[on_best_eval_loss_callback, on_best_bleu2_callback, lr_monitor],
        max_epochs=args.epochs,
        accelerator='cuda' if args.cuda else 'cpu',
        devices=args.num_devices,
        precision=args.precision, # 32 if has more vram
        gradient_clip_val=10.0,
        logger=[wandb_logger],
        accumulate_grad_batches=args.grad_accum,
        deterministic=True,
        check_val_every_n_epoch=2
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader, test_dataloader])

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
    parser.add_argument('--dataset_name', type=str, default='lpm-24')
    parser.add_argument('--model_config', type=str, default='src/configs/config_main.yaml')
    parser.add_argument('--output_folder', type=str, default='ckpt/')
    
    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)
        
    main(args)