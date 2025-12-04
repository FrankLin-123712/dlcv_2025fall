#!/bin/bash

python3 p1_train.py \
    --data_path ./hw2_data/digits \
    --ckpt_path ./ckpts/p1 \
    --gpu 1 \
    --n_feat 256 \
    --n_T 1000 \
    --n_ep 150 \
    --bs 200 \
    --lr 2e-4 \
    --wd 1e-5