from __future__ import division, print_function, absolute_import

import os
import pdb
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class VQAv2(Dataset):

    def __init__(self, root, train, seqlen=14):
        """
        root (str): path to data directory
        train (bool): training or validation
        seqlen (int): maximum words in a question
        """
        if train:
            prefix = 'train'
        else:
            prefix = 'val'
        print("Loading preprocessed files... ({})".format(prefix))
        qas = pickle.load(open(os.path.join(root, prefix + '_qa.pkl'), 'rb'))
        idx2word, word2idx = pickle.load(open(os.path.join(root, 'dict_q.pkl'), 'rb'))
        idx2ans, ans2idx = pickle.load(open(os.path.join(root, 'dict_ans.pkl'), 'rb'))
        vfeats = pickle.load(open(os.path.join(root, prefix + '_vfeats.pkl'), 'rb'))

        print("Setting up everything... ({})".format(prefix))
        self.vqas = []
        for qa in tqdm(qas):
            que = np.ones(seqlen, dtype=np.int64) * len(word2idx)
            for i, word in enumerate(qa['question_toked']):
                if word in word2idx:
                    que[i] = word2idx[word]

            ans = np.zeros(len(idx2ans), dtype=np.float32)
            for a, s in qa['answer']:
                ans[ans2idx[a]] = s

            self.vqas.append({
                'v': vfeats[qa['image_id']],
                'q': que,
                'a': ans,
                'q_txt': qa['question'],
                'a_txt': qa['answer']
            })

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        return self.vqas[idx]['v'], self.vqas[idx]['q'], self.vqas[idx]['a'], self.vqas[idx]['q_txt'], self.vqas[idx]['a_txt']

    @staticmethod
    def get_n_classes(fpath=os.path.join('data', 'dict_ans.pkl')):
        idx2ans, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2ans)

    @staticmethod
    def get_vocab_size(fpath=os.path.join('data', 'dict_q.pkl')):
        idx2word, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2word)


def prepare_data(args):

    train_loader = torch.utils.data.DataLoader(
        VQAv2(root=args.data_root, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=args.pin_mem)

    val_loader = torch.utils.data.DataLoader(
        VQAv2(root=args.data_root, train=False),
        batch_size=args.vbatch_size, shuffle=False, num_workers=args.n_workers, pin_memory=args.pin_mem)

    vocab_size = VQAv2.get_vocab_size()
    num_classes = VQAv2.get_n_classes()
    return train_loader, val_loader, vocab_size, num_classes
