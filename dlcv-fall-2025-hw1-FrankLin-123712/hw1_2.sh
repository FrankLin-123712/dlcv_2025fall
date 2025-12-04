#!/bin/bash

# TODO - run your inference Python3 code
# usage : python3 p2_inference_resnet101_deeplabv3.py <test_image_dir> <output_dir> <model_path> <work_dir>
python3 src/p2_inference_resnet101_deeplabv3.py "$1" "$2" "ckpts_p2.pth" "./"