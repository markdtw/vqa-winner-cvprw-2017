from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from model import Model
from loader import Data_loader

def train(args):

    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available.')

    loader = Data_loader(args.bsize, args.emb)

    model = Model(vocab_size=loader.q_words,
                  emb_dim=loader.emb_dim,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)
    
    criterion = nn.BCELoss()
    
    model = model.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adadelta(model.parameters())

    if args.modelpath:
        if os.path.isfile(args.modelpath):
            print ('Resuming from checkpoint %s' % (args.modelpath))
            ckpt = torch.load(args.modelpath)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])


    for ep in xrange(args.ep):
        ep_loss = 0
        for step in xrange(loader.n_batches):
            # batch preparation
            q_batch, a_batch, i_batch = loader.next_batch()
            q_batch, a_batch, i_batch = Variable(q_batch), Variable(a_batch), Variable(i_batch)
            q_batch, a_batch, i_batch = q_batch.cuda(), a_batch.cuda(), i_batch.cuda()
            optimizer.zero_grad()

            # inference model and optimize
            output = model(q_batch, i_batch)
            loss = criterion(output, a_batch)
            loss.backward()
            optimizer.step()

            # some stats
            ep_loss += loss.data[0]
            if step % 40 == 0:
                print ('Epoch %02d(%04d), loss: %.3f' % (ep+1, step, loss))
        
        # save model after every epoch
        tbs = {
            'epoch': ep + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(tbs, 'model-' + str(ep+1) + '.pth.tar')
        print ('Epoch %02d done, average loss: %.3f' % (ep+1, ep_loss / loader.n_batches))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Winner of VQA 2.0 in CVPR\'17 Workshop')
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int, default=20, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=512, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int, default=512, help='hidden dimension.')
    parser.add_argument('--emb', metavar='', type=int, default=300, help='embedding dimension. (50, 100, 200, *300)')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if not args.train:
        parser.print_help()

