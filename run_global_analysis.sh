#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=global_analysis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 global_analysis.py -modeldir='/usr/project/xtmp/ct214/saved_models/resnet152/5class_DDSM_1024_0506/' -model='60_2push0.7705.pth'