#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=test_vis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python local_analysis.py -test_image DP_AKAY_89024.npy
srun -u python local_analysis.py -test_image DP_AKCX_1886.npy
srun -u python local_analysis.py -test_image DP_AKUT_17603.npy
