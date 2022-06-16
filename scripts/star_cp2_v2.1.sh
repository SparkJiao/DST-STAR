#!/bin/bash

gpu=$1
seed=$2

attn_rank=sssvvs

output_dir=out-bert/exp-cp2-v2.1.s${seed}.A100

CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --model_ver 2 --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed} \
--enc_lr 2e-5 --dec_lr 6e-5 --val_attn_residual