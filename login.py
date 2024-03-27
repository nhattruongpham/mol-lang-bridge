import wandb
import subprocess
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wandb_key', type=str, required=True)
    parser.add_argument('--hf_token', type=str, required=True)
    
    args = parser.parse_args()
    
    wandb.login(key=f'{args.wandb_key}')
    subprocess.run(["huggingface-cli", "login", "--token", f"{args.hf_token}"]) 