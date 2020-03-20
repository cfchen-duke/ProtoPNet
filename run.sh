#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=spiculated_50000

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=1024 -experiment_run="PPNETSpiculated0319_1024"