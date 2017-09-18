from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import json
import cPickle as pickle

import numpy as np

class Data_loader:
    # Before using data loader, make sure your data/ folder contains required files
    def __init__(self, batch_size=512, emb_dim=300, multilabel=False, train=True):
        self.bsize = batch_size
        self.emb_dim = emb_dim
        self.multilabel = multilabel
        self.train = train
        self.seqlen = 14    # hard set based on paper

        q_dict = pickle.load(open('data/train_q_dict.p', 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        a_dict = pickle.load(open('data/train_a_dict.p', 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        if train:
            self.vqa = json.load(open('data/vqa_train_final.json'))
            self.n_questions = len(self.vqa)
        else:
            self.vqa = json.load(open('data/vqa_val_final.json'))
            self.n_questions = len(self.vqa)

        # should have more efficient way to load image feature
        self.i_feat = np.load('data/coco_features.npy').item()

        print ('Loading done')

        # initialize loader
        self.n_batches = self.n_questions // self.bsize
        self.K = self.i_feat.values()[0].shape[0]
        self.feat_dim = self.i_feat.values()[0].shape[1]
        self.init_pretrained_wemb(emb_dim)
        self.epoch_reset()

    def init_pretrained_wemb(self, emb_dim):
        """From blog.keras.io"""
        embeddings_index = {}
        f = open('data/glove.6B.' + str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v
        
        self.pretrained_wemb = embedding_mat

    def epoch_reset(self):
        self.batch_ptr = 0
        np.random.shuffle(self.vqa)

    def next_batch(self):
        """Return 3 things:
        question -> (seqlen, batch)
        answer -> (batch, n_answers) or (batch, )
        image feature -> (batch, feat_dim)
        """
        if self.batch_ptr + self.bsize >= self.n_questions:
            self.epoch_reset()

        q_batch = []
        a_batch = []
        i_batch = []
        for b in xrange(self.bsize):
            # question batch
            q = [0] * self.seqlen
            for i, w in enumerate(self.vqa[self.batch_ptr + b]['question_toked']):
                try:
                    q[i] = self.q_wtoi[w]
                except:
                    q[i] = 0    # validation questions may contain unseen word
            q_batch.append(q)

            # answer or question id batch
            if self.train:
                if self.multilabel:
                    a = np.zeros(self.n_answers, dtype=np.float32)
                    for w, c in self.vqa[self.batch_ptr + b]['answers_w_scores']:
                        a[self.a_wtoi[w]] = c
                    a_batch.append(a)
                else:
                    try:
                        a_batch.append(self.a_wtoi[self.vqa[self.batch_ptr + b]['answer']])
                    except:
                        a_batch.append(0)
            else:
                # in validation phase return question id instead of answer to write to a json file
                # you could also modify this to calculate the accuracy
                a_batch.append(self.vqa[self.batch_ptr + b]['question_id'])
            
            # image batch
            iid = self.vqa[self.batch_ptr + b]['image_id']
            i_batch.append(self.i_feat[iid])

        self.batch_ptr += self.bsize
        q_batch = np.asarray(q_batch)   # (batch, seqlen)
        a_batch = np.asarray(a_batch)   # (batch, n_answers) or (batch, )
        i_batch = np.asarray(i_batch)   # (batch, feat_dim)
        return q_batch, a_batch, i_batch

