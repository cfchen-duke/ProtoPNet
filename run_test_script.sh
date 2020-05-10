#!/bin/bash
#SBATCH  --gres=gpu:1 -p compsci-gpu --job-name=test_Vgg

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python vgg_features.py
