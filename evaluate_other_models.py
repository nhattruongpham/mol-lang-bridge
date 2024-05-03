import selfies as sf
import argparse
import os
import torch
from src.scripts.mydatasets import get_dataloader, Lang2molDataset_2
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

os.system(f"huggingface-cli login --token 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")

def collate(batch):
        input_ids = [i[0] for i in batch]
        attention_mask = [i[1] for i in batch]
        smiles = [i[2] for i in batch]
        selfies = [i[3] for i in batch]
        return (torch.concat(input_ids, dim=0), torch.concat(attention_mask, dim=0), smiles, selfies)

class EvalDataset():
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        caption = sample['caption']
        smiles = sample['canonical']
        selfies = sample['selfies']
        
        task_definition = 'Definition: You are given a molecule description in English. Your job is to generate the molecule SELFIES that fits the description.\n\n'
        task_input = f'Now complete the following example -\nInput: {caption}\nOutput: '
        model_input = task_definition + task_input
        model_input = self.tokenizer(model_input,
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length",
                                   return_attention_mask=True)
        input_ids = model_input.input_ids
        attention_mask = model_input.attention_mask
        
        return input_ids, attention_mask, smiles, selfies
        

def main(args):
    set_seed(42)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_name)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_name)
    
    # if torch.cuda.is_available():
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    #     device = 'cuda'
    # else:
    #     device = 'cpu'
    model = model.to(args.devices)
    model.eval()
    generation_config = model.generation_config
    generation_config.max_length = 512
    generation_config.num_beams = 1
    eval_dataset = EvalDataset(
        load_dataset("ndhieunguyen/LPM-24",
                     use_auth_token=True,
                     split=args.split,
                     ), 
        tokenizer,
        )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate)

    final_result = []
    for input_ids, attention_mask, smiles, selfies in tqdm(eval_dataloader, total=len(eval_dataloader)):
        try:
            outputs = model.generate(input_ids.to(args.devices), attention_mask=attention_mask.to(args.devices), generation_config=generation_config)
            for i, output in enumerate(outputs):
                output = tokenizer.decode(output, skip_special_tokens=True).replace(' ', '')
                if selfies:
                    output = sf.decoder(output)
                final_result.append(output + '\t||\t' + smiles[i])
        except:
            pass
        
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(final_result))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_name', type=str, default='QizhiPei/biot5-base-text2mol')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--selfies', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, default='output.txt')
    parser.add_argument('--devices', default='0')
    args = parser.parse_args()
    
    main(args)