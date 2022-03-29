#!/bin/bash

python_path=/home/mazen/miniconda3/envs/jdt/bin/python

if [ -z "$1" ]
then
    echo "running on any gpu"
    $python_path ./scripts/det_train.py exp=det project=mot
else
    echo "run on gpu $1"
    CUDA_VISIBLE_DEVICES=$1 $python_path ./scripts/det_train.py exp=det project=mot
fi