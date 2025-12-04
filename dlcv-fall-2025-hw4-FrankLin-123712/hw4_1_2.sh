#!/bin/bash

python3 dust3r_inference.py \
    --index_txt_path $1 \
    --gt_npy_path "" \
    --data_root $2 \
    --output_dir "" \
    --eval_mode R \
    --batch_size 8 \
    --num_workers 4 \
    --model_path $4 \
    --use_model Dust3R \
    --save_pose_path $5 \
    --use_original_endpoints \
    --interpolated_dir $3 \
    --test_only