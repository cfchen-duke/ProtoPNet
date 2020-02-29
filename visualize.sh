#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=test_vis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python local_analysis.py -test_image DP_AAPL_R_CC_1#0.npy
srun -u python local_analysis.py -test_image DP_ABBF_R_CC_1#0.npy
srun -u python local_analysis.py -test_image DP_ABDH_R_ML_3#0.npy
