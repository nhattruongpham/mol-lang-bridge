import numpy as np
import os
import torch

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from .text2mol import Text2MolMLP
import nltk
from rdkit import Chem
from tqdm import tqdm

nltk.download("wordnet")
nltk.download("omw-1.4")

class Mol2Text_translation:
    def __init__(self, device='cpu', text_model='allenai/scibert_scivocab_uncased', eval_text2mol=False):
        self.text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        self.eval_text2mol = eval_text2mol
        if eval_text2mol:
            self.text2mol_model = Text2MolMLP(
                ninp=768,
                nhid=600,
                nout=300,
                model_name_or_path=text_model,
                cid2smiles_path=os.path.join(os.path.dirname(__file__), 'assets', 'cid_to_smiles.pkl'),
                cid2vec_path=os.path.join(os.path.dirname(__file__), 'assets', 'test.txt')
            )
            self.device = torch.device(device)
            self.text2mol_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'assets', 'test_outputfinal_weights.320.pt'),
                                                        map_location=self.device), strict=False)
            self.text2mol_model.to(self.device)
        
    def __norm_smile_to_isomeric(self, smi):
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi

    def __call__(self, predictions, references, smiles=None, text_trunc_length=512):
        meteor_scores = []
        text2mol_scores = []

        refs = []
        preds = []

        if self.eval_text2mol:
            zip_iter = zip(references, predictions, smiles)
        else:
            zip_iter = zip(references, predictions)
        
        for t in tqdm(zip_iter):
            if self.eval_text2mol:
                gt, out, smile = t
            else:
                gt, out = t
            
            gt_tokens = self.text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = self.text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            refs.append([gt_tokens])
            preds.append(out_tokens)

            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)
            
            if self.eval_text2mol:
                t2m_score = self.text2mol_model(self.__norm_smile_to_isomeric(smile), out, self.device).detach().cpu().item()
                text2mol_scores.append(t2m_score)

        bleu2 = corpus_bleu(refs, preds, weights=(.5,.5))
        bleu4 = corpus_bleu(refs, preds, weights=(.25,.25,.25,.25))
        
        if self.eval_text2mol:
            text2mol = np.mean(text2mol_scores)

        _meteor_score = np.mean(meteor_scores)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        rouge_scores = []

        refs = []
        preds = []

        for gt, out in zip(references, predictions):

            rs = scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        return {
            "bleu2": bleu2,
            "bleu4": bleu4,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": _meteor_score,
            'text2mol': text2mol if self.eval_text2mol else None
        }