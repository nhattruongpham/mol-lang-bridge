import wandb
import subprocess

wandb.login(key='<wandb_key>')

subprocess.run(["huggingface-cli", "login", "--token", "<hf_token>"]) 
