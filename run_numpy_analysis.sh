#!/bin/bash
#SBATCH -N1 --job-name=numpyanalysis
source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"

srun -u python analyse_numpy.py
