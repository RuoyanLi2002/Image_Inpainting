#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python3 main.py --seed 42 --data_shape 1 16 16 --num_latents 64 --split_intervals 1 2 2 --max_prod_block_conns 8 --load_model True --model_ckpt "model/model_categorical.jpc" \
                --save_img True --random_patch False --num_samples 10 --cache_dir "cache" --save_dir "sample"
