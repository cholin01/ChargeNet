import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from dataset import ProcessedDataset

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold


def initialize_datasets(args, datadir, dataset, data_path, subset=None, splits=None):
  
    
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    
    
    
    
    
    
    
    
    
                 
    datafiles = {'data': data_path}

    
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) if not isinstance(val[0], str) else val for key, val in f.items()}

    
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    
    all_species = _get_species(datasets)

    return args, datasets


def _get_species(datasets):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.

    """
    
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    print(all_species)

    
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    
    if all_species[0] == 0:
        all_species = all_species[1:]

    
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    return all_species
