from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from model import Model
from loader import Data_loader

def test(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(args.bsize, args.emb, args.multilabel, train=False)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))

    model = Model(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)

    model = model.cuda()

    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise SystemExit('Need to provide model path.')

    result = []
    for step in xrange(loader.n_batches):
        # Batch preparation
        q_batch, a_batch, i_batch = loader.next_batch()
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        q_batch, i_batch = q_batch.cuda(), i_batch.cuda()

        # Do one model forward and optimize
        output = model(q_batch, i_batch)
        _, ix = output.data.max(1)
        for i, qid in enumerate(a_batch):
            result.append({
                'question_id': qid,
                'answer': loader.a_itow[ix[i]]
            })

    json.dump(result, open('result.json', 'w'))
    print ('Validation done')

def train(args):
    # Some preparation
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, don\'t do this.')

    print ('Loading data')
    loader = Data_loader(args.bsize, args.emb, args.multilabel)
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tK: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (loader.q_words, args.emb, loader.K, loader.feat_dim,
                args.hid, loader.n_answers))
    print ('Initializing model')

    model = Model(vocab_size=loader.q_words,
                  emb_dim=args.emb,
                  K=loader.K,
                  feat_dim=loader.feat_dim,
                  hid_dim=args.hid,
                  out_dim=loader.n_answers,
                  pretrained_wemb=loader.pretrained_wemb)
    
    if args.multilabel:
        criterion = nn.BCELossWithLogis()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Continue training from saved model
    if args.modelpath and os.path.isfile(args.modelpath):
        print ('Resuming from checkpoint %s' % (args.modelpath))
        ckpt = torch.load(args.modelpath)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Training script 
    print ('Start training.')
    for ep in xrange(args.ep):
        ep_loss = 0
        ep_correct = 0
        for step in xrange(loader.n_batches):
            # Batch preparation
            q_batch, a_batch, i_batch = loader.next_batch()
            q_batch = Variable(torch.from_numpy(q_batch))
            a_batch = Variable(torch.from_numpy(a_batch))
            i_batch = Variable(torch.from_numpy(i_batch))
            q_batch, a_batch, i_batch = q_batch.cuda(), a_batch.cuda(), i_batch.cuda()

            # Do model forward
            output = model(q_batch, i_batch)
            loss = criterion(output, a_batch)

            # Some stats
            _, oix = output.data.max(1)
            if args.multilabel:
                _, aix = a_batch.data.max(1)
            else:
                aix = a_batch.data
            correct = torch.eq(oix, aix).sum()
            ep_correct += correct
            ep_loss += loss.data[0]
            if step % 40 == 0:
                print ('Epoch %02d(%03d/%03d), loss: %.3f, correct: %3d / %d (%.2f%%)' %
                        (ep+1, step, loader.n_batches, loss.data[0], correct, args.bsize, correct * 100 / args.bsize))

            # compute gradient and do optim step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save model after every epoch
        tbs = {
            'epoch': ep + 1,
            'loss': ep_loss / loader.n_batches,
            'accuracy': ep_correct * 100 / (loader.n_batches * args.bsize), 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(tbs, 'save/model-' + str(ep+1) + '.pth.tar')
        print ('Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (ep+1, ep_loss / loader.n_batches, ep_correct * 100 / (loader.n_batches * args.bsize)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Winner of VQA 2.0 in CVPR\'17 Workshop')
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int, default=50, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=512, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int, default=512, help='hidden dimension.')
    parser.add_argument('--emb', metavar='', type=int, default=300, help='embedding dimension. (50, 100, 200, *300)')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained model path.')
    parser.add_argument('--multilabel', metavar='', type=bool, default=False, help='set this to use multilabel.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.eval:
        test(args)
    if not args.train and not args.eval:
        parser.print_help()

