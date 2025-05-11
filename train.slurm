#!/bin/bash
#SBATCH -J RESP_noupdate  #  本次作业的名称
#SBATCH -o ./devir_result/data/RESP_noupdate.out                  #  %j将被作业号替代
#SBATCH -e ./devir_result/data/RESP_noupdate.error                   #  错误日志输出
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -w  gpu12# If you want to specify a computing node, you can write its name here and remove the first #


module load cuda/11.1
source ~/miniconda3/bin/activate
conda activate pignet

#输入要执行的命令

python ./train.py
