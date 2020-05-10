#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=[v100|p100] -p compsci-gpu --job-name=vanilla_noaug

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"
nvidia-smi

srun -u python vanilla_vgg.py -model="vgg13"  \
                              -train_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class/" \
                              -test_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class_test/"\
                              -name="paper_replicate"\
                              -lr="2e-4" \
                              -wd="1e-3"
