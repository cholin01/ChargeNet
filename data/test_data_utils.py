import logging
import os
import random
import shutil
import rdkit
from rdkit import Chem
import rdkit.Chem.rdmolops as rd
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.spatial import distance_matrix
import pickle

from functools import partial
from tqdm import tqdm








def process_sdf_e4(mol, predict=True):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    
    
    
    
    file_name = mol.GetProp('_Name')
    mol_atom_prop = []
    mol_bond_prop = []
    charges = []

    
    num_atoms = torch.tensor(mol.GetNumAtoms())  
    num_bonds = torch.tensor(mol.GetNumBonds())
    src = []
    dst = []
    efeats = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        bond_feat = bond_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        efeats.append(bond_feat)
        src.append(u)
        dst.append(v)
        
    edge_index = [src, dst]
    edge_index = [[row[i] for row in edge_index] for i in range(len(edge_index[0]))]

    
    
    atom_position = torch.tensor(mol.GetConformers()[0].GetPositions())
    
    for atom in mol.GetAtoms():
        atom_prop = atom_features(atom)
        mol_atom_prop.append(atom_prop)
        atom_charge = atom.GetAtomicNum()
        charges.append(atom_charge)
    
    if not predict:
        ddec_charges = [float(mol.GetAtomWithIdx(i).GetProp('molFileAlias')) for i in range(num_atoms)]
        ddec_charges = torch.tensor(ddec_charges, dtype=torch.float)
    
    if not predict:

        molecule = {'filename': file_name, 'mol_atom_prop': mol_atom_prop, 'ddec_charges': ddec_charges, 
                    'charges': charges, 'mol_bond_prop': efeats, 'edge_index': edge_index}
        
    else:
        molecule = {'filename': file_name, 'mol_atom_prop': mol_atom_prop,
                    'charges': charges, 'mol_bond_prop': efeats, 'edge_index': torch.tensor(edge_index)}

    mol_props = {'num_atoms': num_atoms, 'positions': atom_position}

    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) if not isinstance(val, str) else val for key, val in molecule.items()}

    return molecule

def translation(adj_matrix):
    a = np.zeros((67, 67))
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            a[i][j] = adj_matrix[i][j]
    a = torch.tensor(a)
    return a

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',
                                           'Unknown']) +  
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] +
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                          SP3D, Chem.rdchem.HybridizationType.SP3D2]))


def bond_features(bond):
    bt = bond.GetBondType()
    bs = bond.GetStereo()
    return np.array([int(bt == Chem.rdchem.BondType.SINGLE),
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bs == Chem.rdchem.BondStereo.STEREONONE, bs == Chem.rdchem.BondStereo.STEREOANY, bs == Chem.rdchem.BondStereo.STEREOZ,
                     bs == Chem.rdchem.BondStereo.STEREOE,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


def read_coors(datafile):
    T = False
    with open(datafile, "rb") as f:
    
        lines = f.readlines()
    for i, row_line in enumerate(lines):
        row_line = row_line.decode('utf-8').replace("\n", "")  
        if i == 3:
            number = int(lines[i].split()[0])
            start_line = i + 1
            end_line = i + number
            T = True
            break
    if T:
        cb = np.zeros((number, 3))
        j = 0
        for i in range(start_line , end_line + 1):
            lines[i] = lines[i].decode('utf-8').replace("\n", "")
            lines[i] = [lines[i].split()[0] , lines[i].split()[1] , lines[i].split()[2]]
            ca = np.array(lines[i])
            cb[j] = ca
            j = j + 1
        cc = np.average(cb, axis=0)  
        for i in range(0, 3):
            cb[:, i] = cb[:, i] - cc[i]  
        cb = torch.tensor(cb)

        return cb

def read_ddec_charge(datafile):
    resp_charges = []
    T = False
    with open(datafile, "rb") as f:
        
        lines = f.readlines()
    for i, row_line in enumerate(lines):
        row_line = row_line.decode('utf-8').replace("\n", "")  
        if row_line == 'A    1':
            start_line = i - 1
        elif row_line == 'M  END':
            end_line = i - 1
            T = True
            break
    if T:
        num = (end_line - start_line) / 2
        num = int(num)
        cb = np.zeros((num, 1))
        j = 0
        for i in range(start_line + 2, end_line + 1, 2):
            lines[i] = lines[i].decode('utf-8').replace("\n", "")
            ca = np.array(lines[i])
            cb[j] = ca
            j = j + 1
        cb = torch.tensor(cb)
        return cb

def copyFile(fileDir, save_dir):
    train_rate = 0.8
    valid_rate = 0.1

    image_list = os.listdir(fileDir)  
    image_number = len(image_list)
    train_number = int(image_number * train_rate)
    valid_number = int(image_number * valid_rate)
    train_sample = random.sample(image_list, train_number)  
    valid_sample = random.sample(list(set(image_list) - set(train_sample)), valid_number)
    test_sample = list(set(image_list) - set(train_sample) - set(valid_sample))
    sample = [train_sample, valid_sample, test_sample]

    
    for k in range(len(save_dir)):
        
        
        

        if not os.path.isdir(save_dir[k]):
            os.makedirs(save_dir[k])

        for name in sample[k]:
            shutil.copy(os.path.join(fileDir, name),
                        os.path.join(save_dir[k] + '/', name))  


def convert(T):
    
    props = T[0].keys()
    assert all(props == mol.keys() for mol in T), 'All molecules must have same set of properties/keys!'

    
    T = {prop: [mol[prop] for mol in T] for prop in props}

    
    
    
    T = {key: (pad_sequence(val, batch_first=True) if (not isinstance(val[0], str) and val[0].dim() > 0) else val) 
            for key, val in T.items()}
    return T


def prepare_dataset(datadir, dataset,mutiple_dirs=False, subset=None, splits=None, copy=False):

    
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    
    data_splits = {split: os.path.join(
        datadir + '/', split) for split in split_names}  
    datafiles = {'data': os.path.join(datadir, 'data.npz')}

    save_train_dir = data_splits['train']
    save_valid_dir = data_splits['valid']
    save_test_dir = data_splits['test']
    save_dir = [save_train_dir, save_valid_dir, save_test_dir]
    path = os.path.join(datadir, dataset)
    if copy == True:
        copyFile(path, save_dir)
    else:
        pass
    
    data = []
    i = 0
    if mutiple_dirs:
        
        for filename in tqdm(os.listdir(path)):
            
            data_path = os.path.join(path, filename)
            mol = Chem.SDMolSupplier(data_path, removeHs=False)[0]
            if mol != None:
                data.append(process_sdf_e4(mol))

    else:
        mol_list = Chem.SDMolSupplier(path, removeHs=False)
        for mol in mol_list:
            data.append(process_sdf_e4(mol))
            i = i + 1
            print('已完成:', i)
        
    data = convert(data)

    savedir = os.path.join(datadir, 'HDAC2_ligprep_rename.npz')
    np.savez_compressed(savedir, **data)
    print(savedir)
    print('successful')

    return datafiles

def main():
    datadir = '/home/suqun/score_test/HDAC2'
    dataset = 'HDAC2_ligprep_rename.sdf'
    prepare_dataset(datadir=datadir, dataset=dataset, mutiple_dirs=False)


if __name__ == '__main__':
    main()



