#!/bin/bash

# TODO - run your inference Python3 code
python3 p1_inference.py \
    --mode 1 \
    --model_path ./ckpt_p1.pth \
    --save_path $1 \
    --guide_w 2.0