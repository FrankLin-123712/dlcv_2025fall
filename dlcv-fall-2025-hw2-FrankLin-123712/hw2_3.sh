#!/bin/bash

# TODO - run your inference Python3 code
python3 stable-diffusion/p3_inference.py \
  --json_path $1 \
  --input_dir $2 \
  --output_dir $3 \
  --config ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
  --sd_ckpt $4 \
  --control_ckpt ./ckpt_p3.pth \
  --img_size 512 --steps 30 --cfg 7.5 --seed 42