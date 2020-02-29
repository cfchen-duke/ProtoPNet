#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=latent-shape

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=1024 -experiment_run="PPNETLesionOrNot0227_1_1024"