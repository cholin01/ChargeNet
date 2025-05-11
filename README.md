# ChargeNet: E(3) equivariant graph attention network for atomic charge prediction

![Charge图](https://github.com/user-attachments/assets/1484eb9a-e9c9-4ab5-917b-2b221b36c95f)

## How to create environment
```
conda create -n ChargeNet python==3.10
pip install -f requirements.txt
```
## Data Prepartion
```
python test_data_utils.py
```

## How to use our model


If you want to use ChargeNet predict molecule
```
sbatch predict.slurm
```

## How to use train our model
If you want to train ChargeNet. The train.log has been uploaded to ./run_data.

```
python train.py --data_path {npz_data}
or
sbatch train.slurm
```
