#!/bin/bash

gpu=$1
seed=$2

attn_rank=sssvvs
value_sup=0.7

output_dir=out-bert/exp-cp-v4.4.s${seed}.A100

CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed} \
--enc_lr 2e-5 --dec_lr 5e-5 --value_sup ${value_sup}