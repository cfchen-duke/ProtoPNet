#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=global_analysis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python3 global_analysis.py -modeldir='/usr/project/xtmp/ct214/saved_models/vgg16/3class_Lo1136_512_0914_neglogit-1/' \
                                    -model='100_5push0.9161.pth'