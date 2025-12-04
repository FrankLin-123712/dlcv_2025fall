#!/bin/bash

# echo ">> ------ [1] Run without vcd -------"
# python3 p1_inference.py \
#   --model_path liuhaotian/llava-v1.5-7b\
#   --annotation_file hw3_data/p1_data/val.json\
#   --images_root hw3_data/p1_data/images/val\
#   --output outputs/p1_pred_wo_cd.json

echo ">> ------ [2] Run with vcd -------"
python3 p1_inference.py \
  --model_path liuhaotian/llava-v1.5-7b\
  --annotation_file hw3_data/p1_data/val.json\
  --images_root hw3_data/p1_data/images/val\
  --output outputs/p1_pred_w_cd.json\
  --use_cd\
  --temperature 1.0\
  --cd_alpha 1.0\
  --cd_beta 0.1\
  --device cuda:1