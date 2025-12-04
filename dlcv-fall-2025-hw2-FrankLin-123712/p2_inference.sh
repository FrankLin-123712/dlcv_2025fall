#!/bin/bash

# python3 p2_inference.py \
#     --gpu 0 \
#     --mode 1 \
#     --gt ./hw2_data/face/GT \
#     --noise ./hw2_data/face/noise \
#     --model ./hw2_data/face/UNet.pt \
#     --save_path ./output/p2/ \
#     --eta 0.0

python3 p2_inference.py \
    --gpu 0 \
    --mode 2 \
    --gt ./hw2_data/face/GT \
    --noise ./hw2_data/face/noise \
    --model ./hw2_data/face/UNet.pt \
    --save_path ./output/p2/ \
    --eta 0.25

# python3 p2_inference.py \
#     --gpu 0 \
#     --mode 3 \
#     --gt ./hw2_data/face/GT \
#     --noise ./hw2_data/face/noise \
#     --model ./hw2_data/face/UNet.pt \
#     --save_path ./output/p2/ \
#     --eta 0.0