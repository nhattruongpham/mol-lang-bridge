import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

class Mol2Text_translation:
    def __init__(self, text_model='allenai/scibert_scivocab_uncased'):
        self.text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        pass

    def __call__(self, predictions, references, inputs, text_trunc_length=512, tsv_path="tmp.tsv"):
        meteor_scores = []

        refs = []
        preds = []

        for gt, out in zip(references, predictions):

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

        bleu2 = corpus_bleu(refs, preds, weights=(.5,.5))
        bleu4 = corpus_bleu(refs, preds, weights=(.25,.25,.25,.25))

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

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("SELFIES\tground truth\toutput\n")
            for inp, gt, out in zip(inputs, references, predictions):
                f.write(inp + "\t" + gt + "\t" + out + "\n")

        return {
            "bleu2": bleu2,
            "bleu4": bleu4,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": _meteor_score,
        }