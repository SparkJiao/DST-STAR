#!/bin/bash

gpu=$1
seed=$2

attn_rank=ssvvss

output_dir=out-bert/exp-cp-v3.1.s${seed}.2080Ti
CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed} \
--enc_lr 2e-5 --dec_lr 4e-5 --train_batch_size 8 --pretrained_model pretrained-models/bert-base-uncased

