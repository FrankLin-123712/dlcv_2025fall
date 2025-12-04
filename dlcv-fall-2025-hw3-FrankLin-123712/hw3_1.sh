#!/bin/bash

# TODO - run your inference Python3 code
python3 p1_inference.py\
  --annotation_file $1\
  --images_root $2\
  --model_path $3\
  --output $4\
  --use_cd\
  --temperature 1.0\
  --cd_alpha 1.0\
  --cd_beta 0.1