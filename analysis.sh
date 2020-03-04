#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=global_analysis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 global_analysis.py -modeldir='/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0229_512/' -model='100_14push0.9426.pth'