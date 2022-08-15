#!/bin/bash

path=$1
gpu=$2

#for seed in 42 43 44 45 46; do
#for seed in 42 43; do
for seed in 44 45 46; do
  source ${path} ${gpu} ${seed}
done
