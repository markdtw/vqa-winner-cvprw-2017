from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import numpy as np
import torch

from model import Model
from loader import Data_loader

def train(args):
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA winner CVPR 2017')
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=100, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=32, help='batch size.')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)
    if not args.train and not args.eval:
        parser.print_help()

