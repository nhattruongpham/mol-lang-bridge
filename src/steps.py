import torch
import numpy as np
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu


def train(model, dataloader, tokenizer, optimizer, device="cuda"):
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dataloader))
    for i, data in pbar:
        optimizer.zero_grad()
        y = data["selfies_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["caption_ids"].to(device, dtype=torch.long)
        mask = data["caption_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs.loss
        total_loss += loss.detach().float().item()

        pbar.set_description(
            f"{i+1}/{len(dataloader)} ({round(i/len(dataloader)*100,2)}%) - {loss}"
        )

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def validate(model, dataloader, tokenizer, device="cuda", prefix="train"):
    model.eval()
    predictions = []
    actuals = []
    generation_config = model.generation_config
    generation_config.max_length = 512
    generation_config.num_beams = 1
    with torch.no_grad():
        for data in tqdm(dataloader):
            y = data["selfies_ids"].to(device, dtype=torch.long)
            ids = data["caption_ids"].to(device, dtype=torch.long)
            mask = data["caption_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                generation_config=generation_config,
            )

            try:
                preds = [
                    sf.decoder(
                        tokenizer.decode(
                            g,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        ).replace(" ", "")
                    )
                    for g in generated_ids
                ]
                target = [
                    sf.decoder(
                        tokenizer.decode(
                            t,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        ).replace(" ", "")
                    )
                    for t in y
                ]

                predictions.extend(preds)
                actuals.extend(target)
            except:
                continue

    references, hypotheses = [], []
    for pred, actual in zip(predictions, actuals):
        gt_tokens = [c for c in actual]
        out_tokens = [c for c in pred]
        references.append([gt_tokens])
        hypotheses.append(out_tokens)

    bleu_score = corpus_bleu(references, hypotheses)

    levs = []
    num_exact, bad_mols = 0, 0
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        try:
            m_out = Chem.MolFromSmiles(pred)
            m_gt = Chem.MolFromSmiles(actual)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
        except:
            bad_mols += 1

        levs.append(lev(pred, actual))
    exact_match_score = num_exact / (i + 1)
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols / len(predictions)

    return {
        f"{prefix}_bleu": bleu_score,
        f"{prefix}_exact": exact_match_score,
        f"{prefix}_levenshtein": levenshtein_score,
        f"{prefix}_valid": validity_score,
    }
