#!/bin/bash

gpu=$1
seed=$2

attn_rank=vvssss

output_dir=out-bert/exp-cp-v1.1.s${seed}.A100
CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed}



