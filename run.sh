#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=[v100|p100] -p compsci-gpu

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python main.py -latent=512 -experiment_run="3class_Lo1136_512_1007_neglogit-1" \
                        -base="vgg16" \
                        -last_layer_weight=-1\
                        -train_dir="/usr/xtmp/mammo/Lo1136i_3classs_with_fa/train_augmented_5000/" \
                        -push_dir="/usr/xtmp/mammo/Lo1136i_3classs_with_fa/train/" \
                        -test_dir="/usr/xtmp/mammo/Lo1136i_3classs_with_fa/validation/"
                        # -model="/usr/project/xtmp/ct214/saved_models/resnet152/PPNETLesionOrNot0304_512/90_16push0.9035.pth" \
