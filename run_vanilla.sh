#!/bin/bash
#SBATCH --gres=gpu:2 --constraint=v100,p100 -p compsci-gpu --job-name=vanilla

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python vanilla_vgg.py -model="vgg11_bn"  \
                              -train_dir="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_augmented_morer_with_rot/" \
                              -name="20k_vgg11_bn_lre-4_wd_e-3"\
                              -lr="1e-4" \
                              -wd="1e-3"
