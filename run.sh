#!/bin/bash
#SBATCH  --gres=gpu:1 --constraint=v100,p100 -p compsci-gpu --job-name=spiculated_50000

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=32 -experiment_run="PPNETSpiculated0415_spiculated_32_noneg_augmented" -base="vgg16"  \
                        -train_dir="/usr/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_noneg_augmented/" \
                        -push_dir="/usr/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_noneg/" \
                        -test_dir="/usr/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated_noneg/"
                        #-model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
