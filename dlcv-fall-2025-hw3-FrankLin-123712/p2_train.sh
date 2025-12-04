#!/bin/bash

# python3 p2_train.py \
#     --data_dir hw3_data/p2_data \
#     --output_dir ckpt/p2_v3 \
#     --decoder_ckpt hw3_data/p2_data/decoder_model.bin \
#     --epochs 10 \
#     --batch_size 4 \
#     --lr 5e-4 \
#     --weight_decay 1e-6 \
#     --max_text_len 64 \
#     --lora_rank 16 \
#     --lora_alpha 16.0 \
#     --lora_dropout 0.1 \
#     --lora_targets q_proj,v_proj \
#     --mixed_precision \
#     --log

python3 p2_train.py \
    --data_dir hw3_data/p2_data \
    --output_dir ckpt/p2_v4 \
    --decoder_ckpt hw3_data/p2_data/decoder_model.bin \
    --epochs 10 \
    --batch_size 4 \
    --lr 5e-4 \
    --weight_decay 1e-6 \
    --max_text_len 64 \
    --lora_rank 32 \
    --lora_alpha 32.0 \
    --lora_dropout 0.1 \
    --lora_targets q_proj,v_proj \
    --mixed_precision \
    --log
