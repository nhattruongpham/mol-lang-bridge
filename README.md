<h1 align="center">Lang2Mol-Diff</h1>

<!-- ![tgmdlm](pics/tgmdlm.png) -->
<p align="center">
        📝 <a href="https://aclanthology.org/2024.langmol-1.15/">Paper</a>｜🤗 <a href="https://huggingface.co/spaces/ndhieunguyen/Lang2mol-Diff">Demo</a> | 🚩<a href="https://1drv.ms/u/c/9709b85687cc4037/EYFA2lXGhrBKgASd-BZHXhQByey4In_YTlGY0c2VFJF1Ng?e=akcE7b">Checkpoints</a>
</p>

This repository is the official implementation of [`Lang2Mol-Diff`: A Diffusion-Based Generative Model for Language-to-Molecule Translation Leveraging SELFIES Molecular String Representation](https://github.com/nhattruongpham/mol-lang-bridge/)

## Abstract
> De novo molecule generation from textual descriptions presents a significant challenge due to potential issues with molecule validity using SMILES representation and the limitations inherent to autoregressive models. This work proposes a diffusion-based language-to-molecule generative model (Lang2Mol-Diff) using SELFIES representation, which addresses these concerns by leveraging the strengths of two state-of-the-art molecular generative models: BioT5 and TGM-DLM. Specifically, Lang2Mol-Diff employs BioT5 to tokenize the SELFIES representation, circumventing the validity issues associated with SMILES strings, while incorporating a text diffusion mechanism in TGM-DLM to overcome the limitations of autoregressive models in this domain. Performance evaluation on the L+M-24 benchmark dataset demonstrates that Lang2Mol-Diff outperforms all state-of-the-art methods for molecule generation in terms of validity.


## News
- 2024.7.07: Paper was accepted as a poster presentation in [OpenReview](https://openreview.net/forum?id=j9q7lurl7T)
- 2024.6.01: Submitted paper at [Language + Molecules @ ACL 2024 Workshop](https://language-plus-molecules.github.io/)

## Dataset
The [L+M-24-extra dataset's](https://huggingface.co/datasets/language-plus-molecules/LPM-24_train-extra) `split_train` was employed to train the model while the [L+M-24 dataset's](https://huggingface.co/datasets/language-plus-molecules/LPM-24_train) `split_val` was employed to evaluate the model. Details regarding the data preprocessing methodology can be found in the accompanying paper. This process was utilized to construct a Huggingface dataset which contains 2 splits: `train` and `validation`. Each split has the following columns: `id`, `smiles`, `selfies`, and `caption`.

## Dependencies and Installation
```
conda env create -f environment.yaml
conda activate molecule
```

## Training
```
python3 train.py \
        --dataset_name <huggingface dataset> \
        --batch_size <batch size> \
        --lr_anneal_steps <total training steps>
```

## Inferencing
```
python3 inference.py \
        --dataset_name <huggingface dataset> \
        --model_path <path to model checkpoint> \
        --outputdir <text file>
```

## Evaluation
You can follow the steps in https://github.com/language-plus-molecules/LPM-24-Dataset to evaluate the results of the model. 


## References
This code is based on https://github.com/CRIPAC-DIG/tgm-dlm/

## Citation
If you find it useful, please cite:
```
@inproceedings{nguyen-etal-2024-lang2mol,
    title = "{L}ang2{M}ol-Diff: A Diffusion-Based Generative Model for Language-to-Molecule Translation Leveraging {SELFIES} Representation",
    author = "Nguyen, Nguyen  and
      Pham, Nhat Truong  and
      Tran, Duong  and
      Manavalan, Balachandran",
    editor = "Edwards, Carl  and
      Wang, Qingyun  and
      Li, Manling  and
      Zhao, Lawrence  and
      Hope, Tom  and
      Ji, Heng",
    booktitle = "Proceedings of the 1st Workshop on Language + Molecules (L+M 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.langmol-1.15",
    pages = "128--134",
    abstract = "Generating de novo molecules from textual descriptions is challenging due to potential issues with molecule validity in SMILES representation and limitations of autoregressive models. This work introduces Lang2Mol-Diff, a diffusion-based language-to-molecule generative model using the SELFIES representation. Specifically, Lang2Mol-Diff leverages the strengths of two state-of-the-art molecular generative models: BioT5 and TGM-DLM. By employing BioT5 to tokenize the SELFIES representation, Lang2Mol-Diff addresses the validity issues associated with SMILES strings. Additionally, it incorporates a text diffusion mechanism from TGM-DLM to overcome the limitations of autoregressive models in this domain. To the best of our knowledge, this is the first study to leverage the diffusion mechanism for text-based de novo molecule generation using the SELFIES molecular string representation. Performance evaluation on the L+M-24 benchmark dataset shows that Lang2Mol-Diff outperforms all existing methods for molecule generation in terms of validity. Our code and pre-processed data are available at https://github.com/nhattruongpham/mol-lang-bridge/tree/lang2mol/.",
}
```