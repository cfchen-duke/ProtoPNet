#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=test_vis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python local_analysis.py -test_image DP_AFEX_R_CC_1#1.npy
srun -u python local_analysis.py -test_image DP_AFXO_L_CC_2#0.npy
srun -u python local_analysis.py -test_image DP_ADZW_L_CC_2#0.npy
