# Mol2Lang-VLM
The official implementation of paper **"Mol2Lang-VLM: Vision- and Text-Guided Generative Pre-trained Language Models for Advancing Molecule Captioning through Multimodal Fusion"**

## Abstract
> This paper introduces an enhanced method for refining generative pre-trained language models in molecule captioning by utilizing multimodal features to achieve more accurate caption generation. Our approach leverages the encoder and decoder blocks of the Transformer-based architecture by introducing third sub-layers into both of them. Specifically, we insert sub-layers in the encoder that fuse features from SELFIES strings and molecular images, while the decoder fuses features from SMILES strings and their corresponding descriptions. Performance evaluation on the CheBI-20 and L+M-24 benchmark datasets demonstrates the proposed model's superiority, achieving higher accuracy and quality in caption generation compared to existing methods.

## How to use

### 1. Environment preparation
After cloning the repo, run the following command to install required packages:
```zsh
# installing pytorch, recommend vervion 2.1.2 or above
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 

# installing additional packages
pip install -r requirements.txt
```

### 2. Pretrained models
We use these pretrained models for fine-tuning:

- BioT5: [HuggingFace](https://huggingface.co/QizhiPei/biot5-base)
- SwinOCSR: [Kaggle](https://www.kaggle.com/datasets/gogogogo11/moedel)
- ChemBERTa: [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

Except for BioT5 and ChemBERTa which are automatically downloaded when you start training or evaluating, you need to prepare SwinOCSR's checkpoint from the above link (please use `swin_transform_focalloss.pth`), then put it into `weights/`.

### 3. Benchmark datasets
- LPM-24: [HuggingFace](https://huggingface.co/datasets/duongttr/LPM-24-extend)
- CheBI-20: [HuggingFace](https://huggingface.co/datasets/duongttr/chebi-20-new)

Because the datasets are automatically downloaded from HuggingFace, please send access request and login by following command:
```zsh
huggingface-cli login --token '<hf_token>'
```

For some reason, we cannot make it public at this time.

### 3. Training model

#### Reconstruct checkpoint on LPM-24 dataset:

```zsh
python train.py --epochs 20 --batch_size 8 \
                --grad_accum 8 --warmup_ratio 0.05 --lr 3e-5 \
                --dataset_name lpm-24 --model_config src/configs/ config_lpm24_train.yaml \ 
                --cuda
```

#### Reconstruct checkpoint on CheBI-20 dataset:
```zsh
python train.py --epochs 50 --batch_size 8 \
                --grad_accum 32 --warmup_ratio 0.04 --lr 1e-3 \
                --dataset_name chebi-20 --model_config src/configs/ config_chebi20_train.yaml \ 
                --cuda
```

### 4. Evaluating model
#### Checkpoints
| Checkpoints on dataset| Download link |
|---|---|
|LPM-24|[OneDrive](https://1drv.ms/f/c/fa72f5f3c0e55162/Eu0ZV-wkkcJGuvdqKEH4xBcB5dOg_xoIWe7-aK-GTwWa_g?e=00RVcf)|
|CheBI-20|[OneDrive](https://1drv.ms/f/c/fa72f5f3c0e55162/Eu0ZV-wkkcJGuvdqKEH4xBcB5dOg_xoIWe7-aK-GTwWa_g?e=00RVcf)|

#### Evaluate on LPM-24
```zsh
python eval.py --dataset_name lpm-24 \
               --model_config src/configs/config_lpm24_train.yaml \
               --checkpoint_path path/to/ckpt \
               --cuda
```

#### Evaluate on CheBI-20
```zsh
python eval.py --dataset_name chebi-20 
               --model_config src/configs/config_chebi20_train.yaml \
               --checkpoint_path path/to/ckpt \
               --cuda
```

## Citation
If you are interested, please cite to:
```

```