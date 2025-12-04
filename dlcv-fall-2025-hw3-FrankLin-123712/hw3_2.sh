#!/bin/bash

# TODO - run your inference Python3 code
python3 p2_inference.py\
  --checkpoint ckpt_p2.pt\
  --image_dir $1\
  --output $2\
  --decoder_ckpt $3