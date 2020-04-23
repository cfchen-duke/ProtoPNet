#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=v100,p100 -p compsci-gpu --job-name=spiculated_50000

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=256 -experiment_run="thresholdlogits0_spiculated_256_0423" -base="vgg16"  \
                        -train_dir="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_no_neg_augmented_more/" \
                        -push_dir="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_noneg/" \
                        -test_dir="/usr/project/xtmp/mammo/binary_Feb/sorted_by_Fides_ratings/definite/"
                        #-model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
