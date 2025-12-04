#!/bin/bash
# Usage (TA runner):
#   bash hw1_1.sh $CSV $IMG_DIR $OUT_CSV
# Do NOT modify the three positional args. You MAY change --ckpt to your file name.
python3 src/p1_inference.py "$1" "$2" "$3" \
  --ckpt "ckpts_p1.pth" \
  --num-classes 65 \
  --batch-size 128 \
  --num-workers 2