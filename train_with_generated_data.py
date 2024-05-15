import os
import torch
import argparse
from src.improved_diffusion import gaussian_diffusion as gd
from src.improved_diffusion.respace import SpacedDiffusion
from src.improved_diffusion import dist_util
from src.improved_diffusion.transformer_model import TransformerNetModel
from src.improved_diffusion.resample import create_named_schedule_sampler
from src.improved_diffusion.script_util import model_and_diffusion_defaults
from src.improved_diffusion.script_util import add_dict_to_argparser
from src.improved_diffusion.train_util import TrainLoop
from transformers import set_seed
import torch.distributed as dist
import wandb
from src.scripts.mytokenizers import Tokenizer
from src.scripts.mydatasets import get_dataloader, Lang2molDataset_2
import warnings
import torch.multiprocessing as mp


def main_worker(rank, world_size):
    args = create_argparser().parse_args()
    set_seed(42)

    wandb.login(key=args.wandb_token)
    wandb.init(
        project="ACL_Lang2Mol",
        config=args.__dict__,
    )

    dist_util.setup_dist(rank, world_size)
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
    # model.train()

    print("Total params:", sum(p.numel() for p in model.parameters()))
    print(
        "Total trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("Tokenizer vocab length:", len(tokenizer))

    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(args.diffusion_steps)],
        betas=gd.get_named_beta_schedule("sqrt", args.diffusion_steps),
        model_mean_type=(gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE)),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch="transformer",
        training_mode="e2e",
    )

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    print("Loading data...")
    train_dataset = Lang2molDataset_2(
        dir=args.dataset_path,
        tokenizer=tokenizer,
        split="train",
        corrupt_prob=0.0,
        token_max_length=512,
        dataset_name=args.dataset_name,
    )
    dataloader = get_dataloader(train_dataset, args.batch_size, rank, world_size)
    print("Finish loading data")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=None,
        eval_interval=args.eval_interval,
    ).run_loop()
    dist.destroy_process_group()


def create_argparser():
    defaults = dict()
    text_defaults = dict(
        wandb_token="e7ec68f70281e418d89a918a45859f150aef9405",
        batch_size=8,
        cache_mode="no",
        checkpoint_path="checkpoints",
        class_cond=False,
        config="ll",
        config_name="QizhiPei/biot5-base-text2mol",
        dataset_path="dataset",
        diffusion_steps=2000,
        dropout=0.01,
        e2e_train="",
        ema_rate="0.9999",
        emb_scale_factor=1.0,
        eval_interval=2000,
        experiment="random",
        experiment_mode="lm",
        fp16_scale_growth=0.001,
        gradient_clipping=2.4,
        image_size=8,
        in_channel=16,
        learn_sigma=False,
        log_interval=1000,
        logits_mode=1,
        lr=0.00005,
        lr_anneal_steps=1000000,
        microbatch=-1,
        modality="e2e-tgt",
        model_arch="transformer",
        # model_name_or_path="predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None",
        noise_level=0.0,
        noise_schedule="sqrt",
        num_channels=128,
        num_heads=4,
        num_heads_upsample=-1,
        num_res_blocks=2,
        out_channel=16,
        padding_mode="pad",
        predict_xstart=True,
        preprocessing_num_workers=1,
        rescale_learned_sigmas=True,
        rescale_timesteps=True,
        resume_checkpoint="",
        save_interval=50000,
        schedule_sampler="uniform",
        seed=42,
        timestep_respacing="",
        training_mode="e2e",
        use_bert_tokenizer="no",
        use_checkpoint=False,
        use_fp16=False,
        use_kl=False,
        use_scale_shift_norm=True,
        weight_decay=0.0,
        model_in_channels=32,
        model_model_channels=128,
        model_dropout=0.01,
        model_hidden_size=1024,
        model_num_attention_heads=16,
        model_num_hidden_layers=12,
        dataset_name="ndhieunguyen/LPM-24-extra",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    world_size = 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
