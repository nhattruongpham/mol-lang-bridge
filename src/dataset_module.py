from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer

class MultimodalMoleculeCaptioning(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='ndhieunguyen/LPM-24', 
                 split='train',
                 input_max_length=256,
                 output_max_length=256):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True)
        self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '') # remove invalid selfies
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(args.roberta.pretrained_model_name_or_path)
        
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        input = self.tokenizer(
            "<bom>"+sample['selfies']+"<eom>",
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        output = self.tokenizer(
            "<boc>"+sample['caption']+"<eoc>",
            add_special_tokens=True,
            max_length=self.output_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        smiles_input = self.roberta_tokenizer(
            sample['canonical'],
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()
        
        smiles_input_ids = smiles_input['input_ids'].flatten()
        smiles_attention_mask = smiles_input['attention_mask'].flatten()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images': self.image_transform(sample['image'].convert('L').convert('RGB')),
            'smiles_input_ids': smiles_input_ids,
            'smiles_attention_mask': smiles_attention_mask,
            'canonical': sample['canonical'],
            'selfies': sample['selfies'],
            'caption': sample['caption']
        }

def get_dataloaders(args, tokenizer, batch_size=8, num_workers=4, split='train'):
    if split == 'train':
        return DataLoader(
            MultimodalMoleculeCaptioning(
                args,
                tokenizer=tokenizer,
                split='train',
                input_max_length=512,
                output_max_length=512
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
    elif split == 'validation':
        return DataLoader(
            MultimodalMoleculeCaptioning(
                args,
                tokenizer=tokenizer,
                split='validation',
                input_max_length=512,
                output_max_length=512
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True
        )