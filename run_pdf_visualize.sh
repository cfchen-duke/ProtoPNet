#!/bin/bash
#SBATCH  --job-name=test_vis

source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"

srun -u python pdf_visualize.py
