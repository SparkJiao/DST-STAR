#!/bin/bash

gpu=$1

attn_rank=vvssss

#for seed in 42; do
#for seed in 43; do
#for seed in 44; do
#for seed in 45; do
#for seed in 46; do
for seed in 42; do
  output_dir=out-bert/exp-cp-v1.0.s${seed}
  CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --attn_rank ${attn_rank} --save_dir ${output_dir}
done

for seed in 45 46; do
  output_dir=out-bert/exp-cp-v1.0.s${seed}.A100
#  CUDA_VISIBLE_DEVICES=${gpu} python train_STAR_cp.py --attn_rank ${attn_rank} --save_dir ${output_dir} --random_seed ${seed}
  CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --attn_rank ${attn_rank} --save_dir ${output_dir}
done


