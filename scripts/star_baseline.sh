#!/bin/bash

gpu=$1
seed=$2

output_dir=out-bert/exp.s${seed}.A100
CUDA_VISIBLE_DEVICES=${gpu} python train_STAR.py --save_dir ${output_dir} --random_seed ${seed}
