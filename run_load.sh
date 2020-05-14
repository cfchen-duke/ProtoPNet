#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=load_run --constraint=[v100|p100]

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 load_run.py