#!/bin/bash

python stable-diffusion/p3_inference.py \
  --gpu 0 \
  --json_path ./hw2_data/fill50k/testing/prompt.json \
  --input_dir ./hw2_data/fill50k/testing/source \
  --output_dir ./output/p3 \
  --config ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
  --sd_ckpt ./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt \
  --control_ckpt ./ckpt_p3.pth \
  --img_size 512 --steps 30 --cfg 7.5 --seed 42
