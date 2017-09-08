from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import json
import random
import cPickle as pickle

import numpy as np

class Data_loader:
    # Before using data loader, make sure your data/ folder contains required files
    def __init__(self, batch_size=512, train=True):
        self.batch_size = batch_size

        dictionary = pickle.load(open('data/dictionary.p', 'rb'))
        self.itow = dictionary['itow']
        self.wtoi = dictionary['wtoi']
        self.n_words = len(self.itow)

        self.vqa_train = json.load(open('data/vqa_final_train.json'))
        self.n_questions = len(vqa_final_train)

        image_feat = np.load('data/coco_features.npy').item()

        print ('Loading done')

    def create_batches(self):
        self.epoch_reset()
        self.n_batches = int(self.n_questions / self.batch_size)
    def next_batch(self):
        if self.batch_ptr + self.batch_size >= self.n_questions: self.epoch_reset()
        image_feat_batch = []
        ques_batch = []
        ans_batch = []
        # TODO
        self.batch_ptr += self.batch_size
        return np.asarray(image_feat_batch), np.asarray(ques_batch), np.asarray(ans_batch)
    def epoch_reset(self):
        self.batch_ptr = 0
        np.random.shuffle(self.vqa_train)
