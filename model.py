from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import numpy as np

class Model(nn.Module):

    def __init__(self, vocab_size, emb_dim, feat_dim, hid_dim, out_dim):
        super(Model, self).__init__()
        # question encoding
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        # TODO: F.normalize(image_feature)

        # image attention
        self.att_wa = nn.Linear(hid_dim, hid_dim)

        # output classifier
        self.clf_wtext = nn.Linear(emb_dim, out_dim)
        self.clf_wimg = nn.Linear(feat_dim, out_dim)

    def forward(self):



    def gated_tanh(self, x, n):
        """
        Implement the gated hyperbolic tangent non-linear activation
        f_a = input x -> R^m, output y -> R^n
            x: input tensor of dimension m
            n: output dimension n
        """
        W = nn.Linear(x.data.shape[0], n)
        W_prime = nn.Linear(x.data.shape[0], n)
        
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y, g)
        return y
