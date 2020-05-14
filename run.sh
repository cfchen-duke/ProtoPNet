#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=[v100|p100] -p compsci-gpu --job-name=neg0

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=2048 -experiment_run="5class_DDSM_2048_0513_neglogit0" -base="resnet201"  \
                        -train_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class_augmented/" \
                        -push_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class/" \
                        -test_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class_test/"
                        #-model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
