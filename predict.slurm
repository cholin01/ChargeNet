#!/bin/bash
#SBATCH -J resp_extra  #  本次作业的名称
#SBATCH -o ./extra_test/resp_test/predict.out                  #  %j将被作业号替代
#SBATCH -e ./extra_test/resp_test/predict.error                   #  错误日志输出
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=10
##SBATCH --gres=gpu:1
##SBATCH -w  gpu12# If you want to specify a computing node, you can write its name here and remove the first #


# module load cuda/11.1
source ~/miniconda3/bin/activate
conda activate py38

#输入要执行的命令

# RESP
# python -u ./predict.py --ckpt_path /home/suqun/model/EGAT/end_result/ckpt/kfold/RESP1.pth --meann 5.6592e-05 --mad 0.1221 --data_path /home/suqun/score_test/HDAC2/HDAC2_ligprep_rename.npz --datadir /home/suqun/score_test/HDAC2 --input_file HDAC2_ligprep_rename.sdf --output_file HDAC2_ligprep_RESP.npz --output_dir MOL2_RESP

# extra resp
# python -u ./predict.py --ckpt_path ./end_result/ckpt/kfold/RESP1.pth --meann 5.6592e-05 --mad 0.1221 --data_path ./dataset/combined_10.npz --datadir ./dataset --input_file combined_10.sdf --output_dir MOL2_RESP

# DDEC4
python -u ./predict.py --ckpt_path ./end_result/ckpt/kfold/DDEC41.pth --meann 0.000263668 --mad 0.10594 --data_path ./submit_data/part3_element/merged.npz --datadir /home/gouqiaolin/tmp/EGAT/submit_data/part3_element --input_file merged.sdf --output_dir merged_predict

# python -u ./predict.py --ckpt_path /home/suqun/model/EGAT/end_result/ckpt/kfold/DDEC781.pth --meann 0.002475 --mad 0.1111242 --data_path  /home/suqun/score_test/HDAC2/HDAC2_ligprep_rename.npz --datadir /home/suqun/score_test/HDAC2 --input_file HDAC2_ligprep_rename.sdf --output_file HDAC2_ligprep_DDEC78.npz --output_dir MOL2_DDEC78

# DDEC78
# python -u ./predict.py --ckpt_path /home/suqun/model/EGAT/end_result/ckpt/kfold/DDEC781.pth --meann 0.002475 --mad 0.1111242 --data_path  /home/suqun/score_test/MCL1/data.npz --datadir /home/suqun/score_test/MCL1 --input_file glide_active.sdf --output_file MCL1_DDEC78.sdf --output_dir MOL2_DDEC78
