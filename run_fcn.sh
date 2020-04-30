#!/bin/bash
#SBATCH --gres=gpu:2 --constraint=v100,p100 -p compsci-gpu --job-name=vanilla

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python vanilla_fcn.py -model="fcn16"  \
                              -train_dir="/usr/project/xtmp/mammo/binary_Feb/lesion_or_not_augmented_more/" \
                              -test_dir="/usr/project/xtmp/mammo/binary_Feb/lesion_or_not_test/" \
                              -name="30k_fcn16_lre-4_wd_e-3_lesion_or_not"\
                              -lr="1e-4" \
                              -wd="1e-3"
