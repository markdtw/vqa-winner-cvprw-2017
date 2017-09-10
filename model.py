from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, vocab_size, emb_dim, K, feat_dim, hid_dim, out_dim, pretrained_wemb):
        super(Model, self).__init__()
        """
        Args:
            vocab_size: vocabularies to embed (question vocab size)
            emb_dim: GloVe pre-trained dimension -> 300
            K: image bottom-up attention locations -> 36
            feat_dim: image feature dimension -> 2048
            hid_dim: hidden dimension -> 512
            out_dim: multi-label regression output -> (answer vocab size)
            pretrained_wemb: as its name
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.K = K
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        # question encoding
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)

        # image attention
        #self.iembed = nn.Linear(feat_dim, hid_dim)
        #self.att_wa = nn.Linear(hid_dim, hid_dim)
        self.att_wa = nn.Linear(hid_dim, 1)

        # output classifier
        #self.clf_wtext = nn.Linear(emb_dim, out_dim)
        #self.clf_wimg = nn.Linear(feat_dim, out_dim)
        self.clf_w = nn.Linear(hid_dim, out_dim)

        # initialize word embedding layer weight
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))

    def forward(self, question, image):
        """
        question -> shape (seqlen, batch)
        image -> shape (batch, K, feat_dim)
        """
        # question encoding
        emb = self.wembed(question) # (seqlen, batch, emb_dim)
        encoded = self.gru(emb)     # (seqlen, batch, hid_dim)
        qenc = encoded[-1]          # (batch, hid_dim)
        
        # image encoding
        image = F.normalize(image)  # (batch, K, feat_dim)
        #image = self.iembed(image)  # (batch, K, hid_dim)

        # image attention
        qenc_reshape = qenc.repeat(1, self.K).view(-1, self.K, self.hid_dim)    # (batch, K, hid_dim)
        concated = torch.cat((image, qenc_reshape), -1)                         # (batch, K, feat_dim + hid_dim)
        concated = self._gated_tanh(concated, self.hid_dim)                     # (batch, K, hid_dim)

        #a = self.att_wa(concated)  # (batch, K, hid_dim)
        #A = F.softmax(a.view(-1, self.hid_dim)).view(-1, self.K, self.hid_dim)  # (batch, K, hid_dim)
        #v_head = torch.sum(torch.mul(A, image))                                 # (batch, hid_dim)
        a = self.att_wa(concated)                           # (batch, K, 1)
        A = F.softmax(a.squeeze())                          # (batch, K)
        v_head = torch.bmm(A.unsqueeze(1), image).squeeze() # (batch, feat_dim)

        # element-wise question + image
        q = self._gated_tanh(qenc, self.hid_dim)
        v = self._gated_tanh(v_head, self.hid_dim)
        h = torch.mul(q, v)         # (batch, hid_dim)

        # output classifier
        s_head = F.sigmoid(self.clf_w(self._gated_tanh(h, self.hid_dim)))

        return s_head               # (batch, out_dim)

    def _gated_tanh(self, x, n):
        """
        Implement the gated hyperbolic tangent non-linear activation
            x: input tensor of dimension m
            n: output dimension n
        """
        W = nn.Linear(x.size()[-1], n)
        W_prime = nn.Linear(x.size()[-1], n)
        
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y
