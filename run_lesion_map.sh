#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=load_lesion_map

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 ff_lesion_detection.py