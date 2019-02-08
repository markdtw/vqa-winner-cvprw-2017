from __future__ import division, print_function, absolute_import

import os
import pdb
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model import Model
from utils import GOATLogger, save_ckpt, compute_score
from data_loader import prepare_data

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'eval'])
# Hyper-parameters
FLAGS.add_argument('--hid-dim', type=int,
                   help="Hidden dimension for GRU")
FLAGS.add_argument('--batch-size', type=int,
                   help="Batch size")
FLAGS.add_argument('--vbatch-size', type=int,
                   help="Batch size for validation")
FLAGS.add_argument('--epoch', type=int,
                   help="Epochs to train")
# Paths
FLAGS.add_argument('--data-root', type=str,
                   help="Location of data")
FLAGS.add_argument('--resume', type=str,
                   help="Location to resume model")
FLAGS.add_argument('--save', type=str,
                   help="Location to save model")
FLAGS.add_argument('--wemb-init', type=str,
                   help="Location to pretrained wemb")
# Others
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--n-workers', type=int, default=2,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', type=bool, default=False,
                   help="DataLoader pin memory or not")
FLAGS.add_argument('--log-freq', type=int, default=100,
                   help="Logging frequency")
FLAGS.add_argument('--seed', type=int, default=420,
                   help="Random seed")


def evaluate(val_loader, model, epoch, device, logger):
    model.eval()

    batches = len(val_loader)
    for step, (v, q, a, _, _) in enumerate(tqdm(val_loader, ascii=True)):
        v = v.to(device)
        q = q.to(device)
        a = a.to(device)

        logits = model(v, q)
        loss = F.binary_cross_entropy_with_logits(logits, a) * a.size(1)
        score = compute_score(logits, a)

        logger.batch_info_eval(epoch, step, batches, loss.item(), score)

    score = logger.batch_info_eval(epoch, -1, batches)
    return score


def train(train_loader, model, optim, epoch, device, logger):
    model.train()

    batches = len(train_loader)
    start = time.time()
    for step, (v, q, a, _, _) in enumerate(train_loader):
        data_time = time.time() - start

        v = v.to(device)
        q = q.to(device)
        a = a.to(device)

        logits = model(v, q)
        loss = F.binary_cross_entropy_with_logits(logits, a) * a.size(1)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()

        batch_time = time.time() - start
        score = compute_score(logits, a)
        logger.batch_info(epoch, step, batches, data_time, loss.item(), score, batch_time)
        start = time.time()


def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    logger = GOATLogger(args.mode, args.save, args.log_freq)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        args.devices = torch.cuda.device_count()
        args.batch_size *= args.devices
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)

    # Get data
    train_loader, val_loader, vocab_size, num_classes = prepare_data(args)

    # Set up model
    model = Model(vocab_size, args.wemb_init, args.hid_dim, num_classes)
    model = nn.DataParallel(model).to(device)
    logger.loginfo("Parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # Set up optimizer
    optim = torch.optim.Adamax(model.parameters())

    last_epoch = 0
    bscore = 0.0

    if args.resume:
        logger.loginfo("Initialized from ckpt: " + args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        last_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])

    if args.mode == 'eval':
        _ = evaluate(val_loader, model, last_epoch, device, logger)
        return

    # Train
    for epoch in range(last_epoch, args.epoch):
        train(train_loader, model, optim, epoch, device, logger)
        score = evaluate(val_loader, model, epoch, device, logger)
        bscore = save_ckpt(score, bscore, epoch, model, optim, args.save, logger)

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
