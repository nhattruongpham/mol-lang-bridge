from argparse import ArgumentParser
import yaml
from mim.mim_lightning import SwinMaskedImageModel

import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2

import torch

from mim import build_loader_simmim

# # Create the inverse transform
# inverse_transform = T.Normalize(
#     mean=[-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
#     std=[1 / s for s in IMAGENET_DEFAULT_STD]
# )

def main(args):
    swin_mim = SwinMaskedImageModel(args)
    dataloader = build_loader_simmim(args)
    
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_config', type=str, default='/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/src/configs/swin_config.yaml')
    parser.add_argument('--data_path', type=str, default='/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/my_dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grads_accum', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=1.25e-3)
    parser.add_argument('--warmup_lr', type=float, default=2.5e-7)
    parser.add_argument('--min_lr', type=float, default=2.5e-7)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        args.__setattr__(key, value)
   
    main(args)