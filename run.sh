#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=[v100|p100] -p compsci-gpu --job-name=512neg-1

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=2048 -experiment_run="5class_Lo1136_2048_0902_neglogit-1" \
                        -base="resnet152"  \
                        -last_layer_weight=-1\
                        -train_dir="/usr/xtmp/mammo/Lo1136i/train_augmented_5000/" \
                        -push_dir="/usr/xtmp/mammo/Lo1136i/train/" \
                        -test_dir="/usr/xtmp/mammo/Lo1136i/validation/"
                        -model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
