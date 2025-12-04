#!/bin/bash

python3 p1_inference.py \
    --mode 2 \
    --model_path ./ckpt_p1.pth \
    --save_path ./output/p1 \
    --guide_w 2.0
