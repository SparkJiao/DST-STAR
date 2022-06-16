#!/bin/bash

gpu=$1
seed=$2

attn_rank=ssvvss

output_dir=out-bert/exp-cp-v2.2.s${seed}.A100

CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed} \
--enc_lr 1e-5 --dec_lr 4e-5

#CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --attn_rank ${attn_rank} --save_dir ${output_dir}
