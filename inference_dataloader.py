from src.scripts.mytokenizers import Tokenizer
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import set_seed
from src.improved_diffusion import gaussian_diffusion as gd
from src.improved_diffusion.respace import SpacedDiffusion
from src.improved_diffusion import dist_util, logger
from src.improved_diffusion.transformer_model import TransformerNetModel
from src.improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from src.scripts.mydatasets import Lang2molDataset_3, get_dataloader
import selfies as sf
from tqdm import tqdm

os.system(f"huggingface-cli login --token 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")


def collate(batch):
    caption_state = [i["caption_state"] for i in batch]
    caption_mask = [i["caption_mask"] for i in batch]
    caption = [i["caption"] for i in batch]
    return (
        torch.concat(caption_state, dim=0),
        torch.concat(caption_mask, dim=0),
        caption,
    )


def main():
    set_seed(42)
    args = create_argparser().parse_args()

    logger.configure()
    args.sigma_small = True

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

    validation_dataset = Lang2molDataset_3(
        dir=args.dataset_path,
        tokenizer=tokenizer,
        split=args.split,
        corrupt_prob=0.0,
        dataset_name="ndhieunguyen/LPM-24",
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    print("-------------------- DATASET INFO --------------------")
    print(f"Size: {len(validation_dataset)} samples")
    print(f'Sample shape: {validation_dataset[0]["caption_state"].shape}')

    print(f"Use DDIM: {args.use_ddim}")
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    print(f"Batch size: {args.batch_size}")
    all_outputs = []
    all_captions = []

    for caption_state, caption_mask, caption in tqdm(validation_dataloader):
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
        all_captions += caption

    with open(args.outputdir.replace(".txt", "_submission.txt"), "w") as f:
        for i, x in enumerate(all_outputs):
            f.write(sf.decoder(x.replace("<pad>", "").replace("</s>", "")) + "\n")

    with open(args.outputdir.replace(".txt", "_captions.txt"), "w") as f:
        for i, x in enumerate(all_outputs):
            f.write(all_captions[i] + "\n")


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
        num_workers=1,
        emb_scale_factor=1.0,
        clamp="clamp",
        split="validation",
        model_path="checkpoints/PLAIN_ema_0.9999_300000.pt",
        use_ddim=False,
        batch_size=32,
        top_p=1.0,
        outputdir="finetune_pretrained_ckpt_3000.txt",
        diffusion_steps=2000,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
