#!/bin/bash

python main.py --mode "train" \
               --hid-dim 512 \
               --batch-size 512 \
               --vbatch-size 512 \
               --epoch 160 \
               --data-root data/ \
               --save log/ \
               --wemb-init data/glove_pretrained_300.npy \
