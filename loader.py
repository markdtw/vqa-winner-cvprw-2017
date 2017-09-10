from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import json
import cPickle as pickle

import numpy as np

class Data_loader:
    # Before using data loader, make sure your data/ folder contains required files
    def __init__(self, batch_size=512, emb_dim=300, train=True):
        self.bsize = batch_size
        self.emb_dim = emb_dim
        self.seqlen = 14    # hard set based on paper

        q_dict = pickle.load(open('data/train_q_dict.p', 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow)

        a_dict = pickle.load(open('data/train_a_dict.p', 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow)

        self.vqa_train = json.load(open('data/vqa_train_final.json'))
        self.n_questions = len(vqa_train)

        self.i_feat = np.load('data/coco_features.npy').item()

        print ('Loading done')
        # initialize loader
        self.n_batches = self.n_questions // self.bsize
        self.K = self.i_feat.values()[0].shape[1]
        self.feat_dim = self.i_feat.values()[0].shape[2]
        self.init_pretrained_wemb(emb_dim)
        self.epoch_reset()

    def epoch_reset(self):
        self.batch_ptr = 0
        np.random.shuffle(self.vqa_train)

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
            if embedding_v is None:
                embedding_v = np.zeros((emb_dim), dtype=np.float32)

            embedding_mat[i] = embedding_v
        
        self.pretrained_wemb = embedding_mat

    def next_batch(self):
        """Return 3 things:
        question -> (seqlen, batch)
        answer -> (batch, n_answers)
        image feature -> (batch, feat_dim)
        """
        if self.batch_ptr + self.bsize >= self.n_questions:
            self.epoch_reset()

        q_batch = []
        a_batch = []
        i_batch = []
        # TODO
        for b in xrange(self.bsize):
            # question batch
            q = [0] * self.seqlen
            for i, w in enumerate(self.vqa_train[self.batch_ptr + b]['question_toked']):
                q[i] = self.q_wtoi[w]
            q_batch.append(q)

            # answer batch
            a = np.zeros(self.n_answers, dtype=np.float32)
            for w, c in self.vqa_train[self.batch_ptr + b]['answers_w_scores']:
                a[self.a_wtoi[w]] = c
            a_batch.append(a)
            
            # image batch
            iid = self.vqa_train[self.batch_ptr + b]['image_id']
            i_batch.append(self.i_feat[iid])

        self.batch_ptr += self.bsize
        q_batch = np.asarray(q_batch).transpose(1, 0)   # (seqlen, batch)
        a_batch = np.asarray(a_batch)                   # (batch, n_answers)
        i_batch = np.asarray(i_batch)                   # (batch, feat_dim)
        return q_batch, a_batch, i_batch

