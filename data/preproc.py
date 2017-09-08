from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import argparse
import collections

from tqdm import tqdm
from nltk.tokenize import StanfordTokenizer

def processing(q, phase):

    def build_vocab(questions):
        count_thr = 0
        # count up the number of words
        counts = {}
        for row in questions:
            for word in row['question_toked']:
                counts[word] = counts.get(word, 0) + 1
        cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
        print ('top words and their counts:')
        print ('\n'.join(map(str,cw[:20])))

        # print some stats
        total_words = sum(counts.itervalues())
        print ('total words:', total_words)
        bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
        vocab = [w for w,n in counts.iteritems() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print ('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
        print ('number of words in vocab would be %d' % (len(vocab), ))
        print ('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

        return vocab

    vocab = build_vocab(q)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    for row in tqdm(q):



def tokenize_q(qa, phase):
    MyTokenizer = StanfordTokenizer()
    for row in tqdm(qa):
        row['question_toked'] = MyTokenizer.tokenize(row['question'].lower())[:14]
    
    json.dump(qa, open('vqa_' + phase + '_toked.json'))

def combine_qa(questions, annotations, phase):
    # 443757 questions
    data = []
    for i, q in enumerate(questions['questions']):
        row = {}
        # questions
        row['question'] = q['question']
        row['question_id'] = q['question_id']
        row['image_id'] = q['image_id']
        
        # answers
        assert q['question_id'] == annotations[i]['question_id']
        row['answer'] = annotations[i]['multiple_choice_answer']
        
        answers = []
        for ans in annotations[i]['answers']:
            answers.append(ans['answer'])
        row['answers'] = collections.Counter(answers).most_common()
        data.append(row)

    json.dump(data, open('vqa_' + phase + '_combined.json'))

def download_vqa_v2():
    # download input questions (train + val)
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/')

    # download annotations
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/')

    # extract them
    os.system('unzip zip/v2_Questions_Train_mscoco.zip -d raw/')
    os.system('unzip zip/v2_Questions_Val_mscoco.zip -d raw/')
    os.system('unzip zip/v2_Annotations_Train_mscoco.zip -d raw/')
    os.system('unzip zip/v2_Annotations_Val_mscoco.zip -d raw/')

if __name__ == '__main__':
    # First download and extract
    if not os.path.exists('raw'):
        download_vqa_v2()

    # Combine Q and A
    if not os.path.exists('vqa_train_combined.json'):
        train_q = json.load(open('raw/v2_OpenEnded_mscoco_train2014_questions.json'))
        train_a = json.load(open('raw/v2_mscoco_train2014_annotations.json'))
        combine_qa(train_q, train_a, 'train')

        #val_q = json.load(open('raw/v2_OpenEnded_mscoco_val2014_questions.json'))
        #val_a = json.load(open('raw/v2_mscoco_val2014_annotations.json'))
        #combine_qa(val_q, val_a, 'val')

    # Tokenize
    if not os.path.exists('vqa_train_toked.json'):
        train = json.load(open('vqa_train_combined.json'))
        tokenize_q(train, 'train')

        #val = json.load(open('vqa_val_combined.json'))
        #tokenize_q(val, 'val')

    # Build dictionary
    if not os.path.exists('vqa_train_final.json'):
        train = json.load(open('vqa_train_toked.json'))
        processing(train, 'train')
