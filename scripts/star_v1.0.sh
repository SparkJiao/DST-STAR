#!/bin/bash

gpu=$1
seed=$2

output_dir=out-bert/exp-v1.0.s${seed}.A100
CUDA_VISIBLE_DEVICES=${gpu} python train_STAR.py --save_dir ${output_dir} --random_seed ${seed} \
--enc_lr 3e-5 --dec_lr 8e-5

