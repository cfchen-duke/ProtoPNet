#!/bin/bash
#SBATCH -N1 --job-name=numpyanalysis
source /home/home2/ct214/virtual_envs/ml/bin/activate
echo "start running"

srun -u python analyse_numpy.py -image "DP_AGOL_R_MLO_4#0aug52.npy JMAAB_2_RCC_D0#0aug2.npy"
