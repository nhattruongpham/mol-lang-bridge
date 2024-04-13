from argparse import ArgumentParser
import yaml
from mim.mim_lightning import MaskedImageModel
from mim import build_loader_simmim, set_seed
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def main(args):
    set_seed(42) # Set seed to keep deterministic of the training
    
    train_dataloader, val_dataloader = build_loader_simmim(args)
    args.n_iter_per_epoch = len(train_dataloader) // args.grads_accum
    
    swin_mim = MaskedImageModel(args)
    
    ckpt_callback = ModelCheckpoint(
        dirpath='ckpt/',
        filename='ckpt_{eval_loss}',
        save_top_k=3,
        verbose=True,
        monitor='eval_loss',
        mode='min'
    )
    
    wandb_logger = WandbLogger(log_model=False)
    
    trainer = pl.Trainer(
        callbacks=[ckpt_callback],
        max_epochs=args.epochs,
        accelerator='cuda' if args.cuda else 'cpu',
        devices=args.num_devices,
        precision=args.precision,
        gradient_clip_val=10.0,
        logger=[wandb_logger]
    )

    trainer.fit(swin_mim, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader,
                ckpt_path=args.resume_from_checkpoint)
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/src/configs/swin_config.yaml')
    parser.add_argument('--train_data_path', type=str, default='/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/my_dataset')
    parser.add_argument('--valid_data_path', type=str, default='/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/my_dataset')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grads_accum', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=1.25e-3)
    parser.add_argument('--warmup_lr', type=float, default=2.5e-7)
    parser.add_argument('--min_lr', type=float, default=2.5e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--num_devices', type=int, default=1)
    
    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        args.__setattr__(key, value)
   
    main(args)