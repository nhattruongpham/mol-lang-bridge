# mol-lang-bridge
Language and Molecules Translation for Scientific Insight and Discovery

1. Set up the environments, login HF, create weights folder
```
conda create -n ACLMol python=3.9
conda activate ACLMol
sh prepare.sh
```

2. Finetune Biot5
```
python3 finetune_biot5.py --wandb_key e7ec68f70281e418d89a918a45859f150aef9405 --batch_size 8
```