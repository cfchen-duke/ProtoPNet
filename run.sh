#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=v100,p100 -p compsci-gpu --job-name=spiculated_50000

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=64 -experiment_run="PPNETSpiculated0402_20000_spiculated_64" -base="vgg16"  \
                        -train_dir="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_augmented_morer_with_rot/"
                        #-model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
