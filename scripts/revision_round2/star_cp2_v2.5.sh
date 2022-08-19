#!/bin/bash

gpu=$1
seed=$2

attn_rank=sssvvs

#output_dir=out-bert/exp-cp2-v2.0.s${seed}.A100
output_dir=out-bert/exp-cp2-v2.5.s${seed}.A40

CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --model_ver 2 --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed} \
--val_attn_residual --value_sup 0.0 \
--pretrained_model ../pretrained-models/bert-base-uncased
