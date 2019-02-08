from __future__ import division, print_function, absolute_import

import os
import pdb
import time
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GOATLogger:

    def __init__(self, mode, save, log_freq=100):
        self.save_root = save
        self.log_freq = log_freq
        self.stats = {
            'train': {'iter': [], 'loss': [], 'score': []},
            'eval': {'epoch': [], 'loss': [], 'score': []},
            'xaxis': {'train': 'iter', 'eval': 'epoch'}
        }

        if mode == 'train':
            if not os.path.exists(self.save_root):
                os.mkdir(self.save_root)
            filename = os.path.join(self.save_root, 'console.log')
            logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S',
                filename=filename,
                filemode='w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)
            logging.info("Logger created at {}".format(filename))
        else:
            logging.basicConfig(level=logging.INFO,
                format='%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%b-%d %H:%M:%S')

    def batch_info(self, epoch, step, batches, data_time, loss, score, batch_time):
        if (step+1) % self.log_freq == 0 or (step+1) == batches:
            strout = "[{:3d}][{:4d}/{:4d}] ".format(epoch+1, step+1, batches) + \
                "time for data/train: {:6.4f}/{:6.4f}, loss: {:6.4f}, score: {:6.3f}".format(\
                    data_time, batch_time, loss, score)
            self.loginfo(strout)
            self.save_stats('train')

        g_step = step + batches * epoch
        self.stats['train']['iter'].append(g_step)
        self.stats['train']['loss'].append(loss)
        self.stats['train']['score'].append(score)

    def batch_info_eval(self, epoch, step, batches, loss=0, score=0):
        if step == -1:
            score_mean = np.mean(self.stats['eval']['score'][-batches:])
            strout = "[{:3d}]* Evaluation - score: {:.3f} *".format(epoch+1, score_mean)
            self.loginfo(strout)
            self.save_stats('eval')
            return score_mean

        self.stats['eval']['epoch'].append(epoch)
        self.stats['eval']['loss'].append(loss)
        self.stats['eval']['score'].append(score)


    def save_stats(self, phase='train'):
        data = pd.DataFrame(self.stats[phase])
        data.to_csv(os.path.join(self.save_root, 'stats_{}.csv'.format(phase)))
        xaxis = self.stats['xaxis'][phase]

        plt.style.use('seaborn-darkgrid')
        # plot accuracy
        plt.plot(xaxis, 'score', data=data)
        plt.title('Classification Accuracy ({}, {})'.format(phase, self.stats[phase][xaxis][-1]))
        plt.legend()
        plt.savefig(os.path.join(self.save_root, 'accuracy_{}.png'.format(phase)))
        plt.clf()
        # plot loss
        plt.plot(xaxis, 'loss', data=data)
        plt.title('Cross-entropy Loss ({}, {})'.format(phase, self.stats[phase][xaxis][-1]))
        plt.savefig(os.path.join(self.save_root, 'loss_{}.png'.format(phase)))
        plt.clf()

    def logdebug(self, strout):
        logging.debug(strout)
    def loginfo(self, strout):
        logging.info(strout)
    def logbreak(self):
        logging.info("=" * 80)


def compute_score(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    score = (one_hots * labels)
    return score.cpu().numpy().sum() / logits.shape[0]


def save_ckpt(score, bscore, epoch, model, optim, save, logger):
    if not os.path.exists(os.path.join(save, 'ckpts')):
        os.mkdir(os.path.join(save, 'ckpts'))

    torch.save({
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'score': score}, os.path.join(save, 'ckpts', 'model_{}.pth.tar'.format(epoch)))

    if score > bscore:
        if os.path.exists(os.path.join(save, 'best.pth.tar')):
            os.unlink(os.path.join(save, 'best.pth.tar'))
        os.symlink(os.path.join('ckpts', 'model_{}.pth.tar'.format(epoch)),
                   os.path.join(save, 'best.pth.tar'))
        logger.loginfo("* This is the best score so far. *\n")
    return max(score, bscore)
