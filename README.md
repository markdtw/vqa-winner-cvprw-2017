# 2017 VQA Challenge Winner from CVPR'17
Pytorch implementation of [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge by Teney et al](https://arxiv.org/pdf/1708.02711.pdf).

## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm) (visualizing preprocessing progress only)
- [nltk](http://www.nltk.org/install.html) (also need [this](https://nlp.stanford.edu/software/tokenizer.shtml) to tokenize strings)


## Data
- [VQA 2.0](http://visualqa.org/download.html)
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)


## Preparation
- For preprocessing, go to `data/` folder and execute preproc.py directly.
- You'll need to install the Stanford Tokenizer (java lib), follow the instruction in [their page](https://nlp.stanford.edu/software/tokenizer.shtml).
- The tokenizing step may take up to 36 hrs to process only the training questions (I have a Xeon E5 CPU already), you can write a java code to tokenize them if you want, it should be a lot faster. (Since python nltk will call the java binding, I guess this is the performance bottleneck)
- 

## Train
## Notes
## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).

