#!/bin/bash

# python3 p2_inference.py \
#   --checkpoint ckpt/p2_v1/best.pt \
#   --data_dir hw3_data/p2_data \
#   --split val \
#   --output outputs/p2_pred_v1.json

python3 p2_inference.py \
  --checkpoint ckpt/p2_v2/best.pt \
  --data_dir hw3_data/p2_data \
  --split val \
  --output outputs/p2_pred_v2.json