import lightning as pl
from transformers import T5ForConditionalGeneration
from torch import optim
import math

class BioT5Model(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.model = T5ForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path)
        self.tokenizer = tokenizer
        
    def resize_token_embeddings(self, len_embeddings):
        self.model.resize_token_embeddings(len_embeddings)
    
    def forward(self, input_ids, attention_mask, labels=None):
        decoder_input_ids = labels[:, :-1].contiguous()
        decoder_target_ids = labels[:, 1:].clone().detach()
        decoder_target_ids[labels[:, 1:] == self.tokenizer.pad_token_id] = -100
         
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            labels = decoder_target_ids
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss, logits = self(input_ids, attention_mask, labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        loss, logits = self(input_ids, attention_mask, labels)
        
        # generated_ids = self.model.generate(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask,
        #     max_length=512,
        #     num_beams=2,
        #     repetition_penalty=2.5, 
        #     length_penalty=1.0, 
        #     early_stopping=True
        # )
        
        # preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        # gts = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
        
        
        # # # loss, logits = self(input_ids, attention_mask, labels)
        
        # # self.log("val_loss", loss, prog_bar=True, logger=True)
        
        # evaluated_metrics = self.evaluate_func(preds, gts, batch['selfies'])
        # self.log_dict(evaluated_metrics)
        
        self.log('eval_loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-5)
        
        max_iter = self.cfg.epochs * self.cfg.train_data_len
        warmup_steps = int(max_iter * self.cfg.warmup_ratio)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
    def generate(self, input_selfies):
        generation_config = self.model.generation_config
        generation_config.max_length = 512
        generation_config.num_beams = 1
        decoder_start_token_id = self.tokenizer.decode('<soc>')[0]
        
        inputs = self.tokenizer(
            input_selfies,
            return_tensors='pt'
        )
        
        outputs = self.model.generate(
            input_ids = inputs['input_ids'],
            decoder_start_token_id=decoder_start_token_id,
            generation_config=generation_config
        )
        
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        
    
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)