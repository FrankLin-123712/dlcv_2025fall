#!/bin/bash

python3 stable-diffusion/p3_train.py \
  --gpu 0 \
  --data_path ./hw2_data/fill50k/training \
  --ckpt_path ./ckpts/p3 \
  --config ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
  --sd-ckpt  ./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt \
  --ep 10 \
  --bs 2 \
  --worker 8 \
  --lr 5e-4 \
  --wd 1e-5 \
  --grad_clip 1.0