#!/bin/bash
#SBATCH -J RESP  
#SBATCH -o {log}                  
#SBATCH -e {err}
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
##SBATCH -w  {node_id} # If you want to specify a computing node, you can write its name here and remove the first #

source ~/miniconda3/bin/activate
conda activate ChargeNet

python ./train.py
