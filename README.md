# 2017 VQA Challenge Winner (CVPR'17 Workshop)
Pytorch implementation of [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge by Teney et al](https://arxiv.org/pdf/1708.02711.pdf).

![Model architecture](https://i.imgur.com/phBHIqZ.png)

## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm) (visualizing preprocessing progress only)
- [nltk](http://www.nltk.org/install.html) (and [this](https://nlp.stanford.edu/software/tokenizer.shtml) to tokenize questions)


## Data
- [VQA 2.0](http://visualqa.org/download.html)
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)


## Preparation
- For questions and answers, go to `data/` folder and execute `preproc.py` directly.
- You'll need to install the Stanford Tokenizer, follow the instructions in [their page](https://nlp.stanford.edu/software/tokenizer.shtml).
- The tokenizing step may take up to 36 hrs to process the training questions (I have a Xeon E5 CPU already), write a pure java code to tokenize them should be a lot faster. (Since python nltk will call the java binding, and python is slow)
- For image feature, slightly modify [this code](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/read_tsv.py) to convert tsv to a npy file `coco_features.npy` that contains a list of dictionaries with key being image id and value being the feature (shape: 36, 2048).
- Download and extract GloVe to `data/` folder as well.
- Now we should be able to train, reassure that the `data/` folder should now contain at least:
  ```
  - glove.6B.300d.txt
  - vqa_train_final.json
  - coco_features.npy
  - train_q_dict.p
  - train_a_dict.p
  ```


## Train
Use default parameters:
```bash
python main.py --train
```
Train from a previous checkpoint:
```bash
python main.py --train --modelpath=/path/to/saved.pth.tar
```
Check out tunable parameters:
```bash
python main.py
```

## Evaluate
```bash
python main.py --eval
```
This will generate `result.json` (validation set only), format is referred to [vqa evaluation format](http://www.visualqa.org/evaluation.html). 


## Notes
- The default classifier is softmax classifier, sigmoid multi-label classifier is also implemented but I can't train based on that.
- Training for 50 epochs reach around 64.42% training accuracy.
- For the output classifier, I did not use the pretrained weight since it's hard to retrieve so I followed *eq. 5* in the paper.
- To prepare validation data you need to uncomment some line of code in `data/preproc.py`.
- `coco_features.npy` is a really fat file (34GB including train+val image features), you can split it and modify the data loading mechanisms in `loader.py`.
- This code is tested with train = train and eval = val, no test data included.
- Issues are welcome!


## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).
