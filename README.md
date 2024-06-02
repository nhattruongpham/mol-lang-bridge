<h1 align="center">Lang2Mol-Diff</h1>

<!-- ![tgmdlm](pics/tgmdlm.png) -->
<p align="center">
        📝 <a href="">Paper</a>｜🤗 <a href="">Demo</a> | 🚩<a href="">Checkpoints</a>
</p>

This repository is the official implementation of [`Lang2Mol-Diff`: A Diffusion-Based Generative Model for Language-to-Molecule Translation Leveraging SELFIES Molecular String Representation](https://github.com/nhattruongpham/mol-lang-bridge/)

## News
- 2024.6.1: Submitted paper at [Language + Molecules @ ACL 2024 Workshop](https://language-plus-molecules.github.io/)

## Dataset
The [L+M-24-extra dataset](https://huggingface.co/datasets/language-plus-molecules/LPM-24_train-extra) was employed to train the model. Details regarding the data preprocessing methodology can be found in the accompanying paper. This process was utilized to construct a Huggingface dataset, featuring the following columns: `id`, `smiles`, `selfies`, and `caption`.

## Dependencies and Installation
```
conda env create -f environment.yaml
conda activate molecule
```

## Training
```
python3 train.py
```

## Inferencing
```
python3 inference.py
```

## Evaluation
You can follow the steps in https://github.com/language-plus-molecules/LPM-24-Dataset to evaluate the results of the model. 


## References
This code is based on https://github.com/CRIPAC-DIG/tgm-dlm/

## Citation
If you find it useful, please cite:
```
```