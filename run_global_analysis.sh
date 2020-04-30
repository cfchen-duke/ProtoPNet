#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=global_analysis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 global_analysis.py -modeldir='/usr/project/xtmp/ct214/saved_models/vgg16/thresholdlogits0_spiculated_256_0423/' -model='100_6push0.5750.pth'