#!/bin/bash
#SBATCH -J resp  
#SBATCH -o {log}
#SBATCH -e {error}
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=10
##SBATCH --gres=gpu:1
##SBATCH -w  {node_id}  # If you want to specify a computing node, you can write its name here and remove the first #

source ~/miniconda3/bin/activate
conda activate ChargeNet

#输入要执行的命令

# RESP
python -u ./predict.py --ckpt_path /home/suqun/model/EGAT/end_result/ckpt/kfold/RESP1.pth --meann 5.6592e-05 --mad 0.1221 --data_path /home/suqun/score_test/HDAC2/HDAC2_ligprep_rename.npz --datadir /home/suqun/score_test/HDAC2 --input_file HDAC2_ligprep_rename.sdf --output_file HDAC2_ligprep_RESP.npz --output_dir MOL2_RESP
# DDEC4
# python -u ./predict.py --ckpt_path ./end_result/ckpt/kfold/DDEC41.pth --meann 0.000263668 --mad 0.10594 --data_path ./submit_data/part3_element/merged.npz --datadir /home/gouqiaolin/tmp/EGAT/submit_data/part3_element --input_file merged.sdf --output_dir merged_sdf
# DDEC78
# python -u ./predict.py --ckpt_path /home/suqun/model/EGAT/end_result/ckpt/kfold/DDEC781.pth --meann 0.002475 --mad 0.1111242 --data_path  /home/suqun/score_test/MCL1/data.npz --datadir /home/suqun/score_test/MCL1 --input_file glide_active.sdf --output_file MCL1_DDEC78.sdf --output_dir MOL2_DDEC78
