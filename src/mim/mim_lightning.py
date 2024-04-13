from typing import Any
from lightning import LightningModule
from backbones import build_mim_model
from torch import optim
from mim.lr_scheduler import build_scheduler
from torchvision.utils import make_grid
from utils import inverse_images
import cv2
import torch

class MaskedImageModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = build_mim_model(args)
        self.args = args
        
    def training_step(self, batch, batch_idx):
        image, mask, _ = batch
        _, _, loss = self.model.forward(image, mask)
        self.log('train_loss', loss.item(), prog_bar=True, logger=True)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask, _ = batch
        image_rec, upscale_mask, loss = self.model.forward(image, mask)
        self.log('eval_loss', loss.item(), prog_bar=True, logger=True)
        
        
        if batch_idx == 0:
            unnorm_rec_images = inverse_images(image_rec)
            ori_images = inverse_images(image)
            upscale_mask = upscale_mask.squeeze(1)
            upscale_mask = torch.stack([upscale_mask, upscale_mask, upscale_mask], dim=1)
            
            masked_images = ori_images * ((upscale_mask+1)%2)
            masked_regions = unnorm_rec_images * upscale_mask
            
            stacked_images = []
            for i in range(min(5, len(masked_images))):
                stacked_images.append(
                    torch.hstack([ori_images[i].permute(1,2,0), masked_images[i].permute(1,2,0), masked_regions[i].permute(1,2,0)]).permute(2,0,1)
                )
            
            grid = make_grid(stacked_images)
            self.logger.log_image('generated_images', [grid])
        
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            eps=1e-8,
            betas=(0.9, 0.999),
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay
        )
        
        lr_scheduler = build_scheduler(self.args, optimizer)
        
        return [optimizer], [{
          'scheduler': lr_scheduler,
          "interval": "step",
          "name": "learning_rate"  
        }]