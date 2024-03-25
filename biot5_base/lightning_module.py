import lightning as pl
from transformers import T5ForConditionalGeneration
from torch import optim

class BioT5Model(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss, logits = self(input_ids, attention_mask, labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        loss, logits = self(input_ids, attention_mask, labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=3e-5)