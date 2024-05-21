import lightning as pl
from backbones.lang.t5 import T5ForMultimodalConditionalGeneration
from transformers import RobertaModel
from backbones.vision.swin import SwinTransformer
import torch
from torch import optim
import math

class T5MultimodalModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Initialize multimodal text-based model
        self.t5_model = T5ForMultimodalConditionalGeneration.from_pretrained(
            args.t5.pretrained_model_name_or_path,
            n_attention_heads=args.multimodal.n_attention_heads,
            use_visual_feature=args.multimodal.use_visual_feature,
            use_smiles_feature=args.multimodal.use_smiles_feature,
            use_forget_gate=args.multimodal.use_forget_gate,
            text_feature_dim=args.multimodal.text_feature_dim,
            visual_feature_dim=args.multimodal.visual_feature_dim,
            smiles_feature_dim=args.multimodal.smiles_feature_dim,
            intermidate_dim=args.multimodal.intermediate_dim,
            fusion_encoder_layers=args.multimodal.fusion_encoder_layers,
            fusion_decoder_layers=args.multimodal.fusion_decoder_layers
        )
        
        if args.multimodal.use_visual_feature:
            # Initialize visual model
            self.swin_model = SwinTransformer(
                img_size=args.swin.img_size,
                num_classes=0,
                embed_dim=args.swin.embed_dim,
                depths=args.swin.depths,
                num_heads=args.swin.num_heads
            )
        
            self.swin_model.load_state_dict(
                torch.load(args.swin.pretrained_model_path)['encoder']
            )
        
            self.swin_model.eval()
            for p in self.swin_model.parameters():
                p.requires_grad = False
        
        if args.multimodal.use_smiles_feature:
            # Initialize text model
            self.roberta_model = RobertaModel.from_pretrained(args.roberta.pretrained_model_name_or_path)
            self.roberta_model.eval()
            for p in self.roberta_model.parameters():
                p.requires_grad = False
        
        
    def resize_token_embeddings(self, len_embeddings):
        self.t5_model.resize_token_embeddings(len_embeddings)
    
    def forward(self, input_ids, 
                attention_mask, 
                labels=None, 
                image_features=None, 
                smiles_features=None,
                smiles_attention_mask=None):
        labels[labels == self.args.tokenizer.pad_token_id] = -100
         
        output = self.t5_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            image_features=image_features,
            smiles_features=smiles_features,
            smiles_attention_mask=smiles_attention_mask
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        smiles_input_ids = batch['smiles_input_ids']
        smiles_attention_mask = batch['smiles_attention_mask']
        labels = batch['labels']
        images = batch['images']
        
        image_features = None
        smiles_features = None
        
        with torch.no_grad():
            if self.args.multimodal.use_visual_feature:
                image_features = self.swin_model.forward_features(images, avgpool=False)
            
            if self.args.multimodal.use_smiles_feature:
                smiles_features = self.roberta_model(input_ids=smiles_input_ids, 
                                                    attention_mask=smiles_attention_mask).last_hidden_state
        
        loss, logits = self(input_ids, attention_mask, labels, image_features, smiles_features, smiles_attention_mask)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        smiles_input_ids = batch['smiles_input_ids']
        smiles_attention_mask = batch['smiles_attention_mask']
        labels = batch["labels"]
        images = batch['images']
        
        image_features = None
        smiles_features = None
        
        with torch.no_grad():
            if self.args.multimodal.use_visual_feature:
                image_features = self.swin_model.forward_features(images, avgpool=False)
            
            if self.args.multimodal.use_smiles_feature:
                smiles_features = self.roberta_model(input_ids=smiles_input_ids, 
                                                    attention_mask=smiles_attention_mask).last_hidden_state
        
        loss, logits = self(input_ids, attention_mask, labels, image_features, smiles_features, smiles_attention_mask)
        
        self.log('eval_loss', loss, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        
        max_iter = self.args.epochs * self.args.train_data_len
        warmup_steps = int(max_iter * self.args.warmup_ratio)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
    def generate_captioning(self, inputs,
                            max_length = 512,
                            num_beams= 1,
                            decoder_start_token_id=0,
                            eos_token_id=1,
                            pad_token_id=0):
        
        input_ids = inputs["input_ids"]
        smiles_input_ids = inputs['smiles_input_ids']
        smiles_attention_mask = inputs['smiles_attention_mask']
        images = inputs['images']
        
        image_features = None
        smiles_features = None
        with torch.no_grad():
            if self.args.multimodal.use_visual_feature:
                image_features = self.swin_model.forward_features(images, avgpool=False)
            
            if self.args.multimodal.use_smiles_feature:
                smiles_features = self.roberta_model(input_ids=smiles_input_ids, 
                                                    attention_mask=smiles_attention_mask).last_hidden_state
        
        outputs = self.t5_model.generate(
            input_ids = input_ids,
            image_features=image_features,
            smiles_features=smiles_features,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
        
        return outputs
        
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)