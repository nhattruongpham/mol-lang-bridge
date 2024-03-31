from typing import Any
from lightning import LightningModule
from backbones import build_mim_model
from torch import optim

class SwinMaskedImageModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = build_mim_model(args)
        self.args = args
        
    def training_step(self, batch, batch_idx):
        image, mask = batch
        loss = self.model.forward(image, mask)
        self.log('train_loss', loss.item(), prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask = batch
        loss = self.model.forward(image, mask)
        self.log('eval_loss', loss.item(), prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            eps=1e-8,
            betas=(0.9, 0.999),
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay
        )
        return 