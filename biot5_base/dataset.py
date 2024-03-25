from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class Mol2TextDataset(Dataset):
    def __init__(self, tokenizer, 
                 dataset_name_or_path='ndhieunguyen/LPM-24', 
                 split='train',
                 input_max_length=256,
                 output_max_length=256):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split)
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        input = self.tokenizer(
            sample['selfies'],
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        output = self.tokenizer(
            sample['caption'],
            add_special_tokens=True,
            max_length=self.output_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'selfies': sample['selfies'],
            'caption': sample['caption']
        }

def get_dataloaders(tokenizer, batch_size=8, num_workers=4, split='train'):
    if split == 'train':
        return DataLoader(
            Mol2TextDataset(
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
            Mol2TextDataset(
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