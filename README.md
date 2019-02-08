# 2017 VQA Challenge Winner (CVPR'17 Workshop)
pytorch implementation of [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge by Teney et al](https://arxiv.org/pdf/1708.02711.pdf).

![Model architecture](https://i.imgur.com/phBHIqZ.png)

## Prerequisites
- python 3.6+
- numpy
- [pytorch](http://pytorch.org/) 0.4
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [nltk](http://www.nltk.org/install.html)
- [pandas](https://pandas.pydata.org/)


## Data
- [VQA 2.0](http://visualqa.org/download.html)
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)


## Preparation
- To download and extract vqav2, glove, and pretrained visual features:
  ```bash
  bash scripts/download_extract.sh
  ```
- To prepare data for training:
  ```bash
  python scripts/preproc.py
  ```
- The structure of `data/` directory should look like this:
  ```
  - data/
    - zips/
      - v2_XXX...zip
      - ...
      - glove...zip
      - trainval_36.zip
    - glove/
      - glove...txt
      - ...
    - v2_XXX.json
    - ...
    - trainval_resnet...tsv
    (The above are files created after executing scripts/download_extract.sh)
    - tokenizers/
      - ...
    - dict_ans.pkl
    - dict_q.pkl
    - glove_pretrained_300.npy
    - train_qa.pkl
    - val_qa.pkl
    - train_vfeats.pkl
    - val_vfeats.pkl
    (The above are files created after executing scripts/preproc.py)
  ```

## Train
Use default parameters:
```bash
bash scripts/train.sh
```

## Notes
- Huge re-factor (especially data preprocessing), tested based on pytorch 0.4.1 and python 3.6
- Training for 20 epochs reach around 50% training accuracy. (model seems buggy in my implementation)
- After all the preprocessing, `data/` directory may be up to 38G+
- Some of `preproc.py` and `utils.py` are based on [this repo](https://github.com/hengyuan-hu/bottom-up-attention-vqa)


## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).
