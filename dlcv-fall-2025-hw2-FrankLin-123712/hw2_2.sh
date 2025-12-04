#!/bin/bash

# TODO - run your inference Python3 code
python3 p2_inference.py \
    --mode 1 \
    --noise $1 \
    --save_path $2 \
    --model $3 \
    --eta 0.0