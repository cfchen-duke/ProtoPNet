#!/bin/bash
#SBATCH  --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=spiculated_50000

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=64 -experiment_run="thresholdlogits25_lesion_512_0419" -base="vgg16"  \
                        -train_dir="/usr/project/xtmp/mammo/binary_Feb/lesion_or_not_augmented_more/" \
                        -push_dir="/usr/project/xtmp/mammo/binary_Feb/lesion_or_not/" \
                        -test_dir="/usr/project/xtmp/mammo/binary_Feb/lesion_or_not_test/"
                        #-model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
