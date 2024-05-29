from src.scripts.mytokenizers import Tokenizer
import argparse
import os, json
import torch
from transformers import set_seed
from src.improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from src.improved_diffusion import gaussian_diffusion as gd
from src.improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from src.improved_diffusion import dist_util, logger
from src.improved_diffusion.transformer_model import TransformerNetModel
from src.improved_diffusion import dist_util, logger
from functools import partial
from src.improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from src.scripts.mydatasets import Lang2molDataset_eval
import os
import selfies as sf
from tqdm import tqdm

os.system(f"huggingface-cli login --token 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")


def main():
    set_seed(42)
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == "random1":
        args.experiment = "random"
    logger.log("creating model and diffusion...")
    tokenizer = Tokenizer()
    model = TransformerNetModel(
        in_channels=args.model_in_channels,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        vocab_size=len(tokenizer),
        hidden_size=args.model_hidden_size,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers=args.model_num_hidden_layers,
    )
    model.eval()
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0, args.diffusion_steps, 10)],
        betas=gd.get_named_beta_schedule("sqrt", args.diffusion_steps),
        model_mean_type=(gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE)),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch="transformer",
        training_mode="e2e",
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    print("--" * 30)
    print(f"Loading {args.split} set")
    print("--" * 30)

    validation_dataset = Lang2molDataset_eval(
        dir=args.dataset_path,
        tokenizer=tokenizer,
        split=args.split,
        corrupt_prob=0.0,
        token_max_length=args.token_max_length,
        dataset_name="ndhieunguyen/LPM-24",
    )
    print("-------------------- DATASET INFO --------------------")
    print(f"Size: {len(validation_dataset)} samples")
    print(f'Sample shape: {validation_dataset[0]["caption_state"].shape}')

    print(f"Use DDIM: {args.use_ddim}")
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    print(f"Batch size: {args.batch_size}")
    next_batch_start = args.start
    next_batch_end = next_batch_start + args.batch_size
    all_outputs = []
    all_caption = []
    all_canonical = []
    pbar = tqdm(
        total=len(validation_dataset) // args.batch_size + 1
        if len(validation_dataset) % args.batch_size != 0
        else len(validation_dataset) // args.batch_size
    )
    while True:
        sample = [
            (
                validation_dataset[i]["caption_state"],
                validation_dataset[i]["caption_mask"],
                validation_dataset[i]["caption"],
                validation_dataset[i]["canonical"],
            )
            for i in range(next_batch_start, next_batch_end)
        ]
        caption_state = torch.concat([i[0] for i in sample], dim=0)
        caption_mask = torch.concat([i[1] for i in sample], dim=0)
        caption = [i[2] for i in sample]
        canonical = [i[3] for i in sample]

        outputs = sample_fn(
            model,
            (args.batch_size, 256, model.in_channels),
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs={},
            top_p=args.top_p,
            progress=True,
            caption=(caption_state, caption_mask),
        )

        # outputs = torch.concat(outputs, dim=0)
        logits = model.get_logits(torch.tensor(outputs).cuda())  # bsz, seqlen, vocab
        cands = torch.topk(logits, k=1, dim=-1)
        outputs = cands.indices
        outputs = outputs.squeeze(-1)
        outputs = tokenizer.decode(outputs)

        all_outputs += outputs
        all_caption += caption
        all_canonical += canonical

        next_batch_start = next_batch_end
        next_batch_end = min(next_batch_end + args.batch_size, len(validation_dataset))
        pbar.update(1)

        if next_batch_start == len(validation_dataset):
            break

    with open(args.outputdir.replace(".txt", "_1.txt"), "w") as f:
        for i, x in enumerate(all_outputs):
            f.write(
                sf.decoder(x.replace("<pad>", "").replace("</s>", ""))
                + "   ||   "
                + all_canonical[i]
                + "\n"
            )

    with open(args.outputdir.replace(".txt", "_2.txt"), "w") as f:
        for i, x in enumerate(all_outputs):
            f.write(
                all_caption[i]
                + "\t"
                + all_canonical[i]
                + "\t"
                + sf.decoder(x.replace("<pad>", "").replace("</s>", ""))
                + "\n"
            )


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        mbr_sample=1,
        model_path="",
        model_arch="conv-unet",
        verbose="yes",
    )
    text_defaults = dict(
        modality="text",
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        dataset_path="dataset",
        experiment="gpt2_pre_compress",
        model_arch="trans-unet",
        model_in_channels=32,
        model_model_channels=128,
        model_dropout=0.1,
        model_hidden_size=1024,
        model_num_attention_heads=16,
        model_num_hidden_layers=12,
        preprocessing_num_workers=1,
        emb_scale_factor=1.0,
        clamp="clamp",
        split="validation",
        model_path="checkpoints/PLAIN_ema_0.9999_500000.pt",
        use_ddim=False,
        batch_size=2,
        top_p=1.0,
        outputdir="output.txt",
        diffusion_steps=2000,
        token_max_length=256,
        start=0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
