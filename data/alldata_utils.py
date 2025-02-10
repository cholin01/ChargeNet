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
import dgl
from scipy.spatial import distance_matrix
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    atom_chiral_tag_one_hot, one_hot_encoding, bond_is_conjugated, atom_formal_charge, atom_num_radical_electrons, \
    bond_is_in_ring, bond_stereo_one_hot
import pickle
from dgllife.utils import BaseBondFeaturizer
from dgl.data.utils import save_graphs, load_graphs

from functools import partial





def chirality(atom):  
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'H',
                                                                                        'Si'], encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    
    ab = b - a  
    ac = c - a  
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_



def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()






def process_sdf_e4(datafile):
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
    file_name = os.path.basename(datafile).split('.')[0]
    mol = Chem.MolFromMolFile(datafile, removeHs=False)
    mol_atom_prop = []
    mol_bond_prop = []
    charges = []

    g = dgl.DGLGraph()  

    
    num_atoms = torch.tensor(mol.GetNumAtoms())  
    g.add_nodes(num_atoms)
    num_bonds = torch.tensor(mol.GetNumBonds())
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = torch.tensor(np.concatenate([src, dst]))
    dst_ls = torch.tensor(np.concatenate([dst, src]))

    g.add_edges(src_ls, dst_ls)

    efeats = torch.tensor(BondFeaturizer(mol)['e'])  
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    
    

    g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
    g.ndata['pos'] = torch.tensor(mol.GetConformers()[0].GetPositions())

    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)

    for atom in mol.GetAtoms():
        atom_prop = atom_features(atom)
        mol_atom_prop.append(atom_prop)
        atom_charge = atom.GetAtomicNum()
        charges.append(atom_charge)
    atom_position = read_coors(datafile)
    atom_ddec_charge = read_ddec_charge(datafile)
    atom_ddec_charge = atom_ddec_charge.squeeze(-1)

    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)

    efeats = g.edata.pop('e')

    efeats = torch.tensor([item.cpu().detach().numpy() for item in efeats])

    edge_index = g.edges()

    edge_index = [[row[i] for row in edge_index] for i in range(len(edge_index[0]))]

    molecule = {'filename': file_name, 'mol_atom_prop': mol_atom_prop, 'charges': charges, 'mol_bond_prop': efeats,
                'edge_index': edge_index}

    mol_props = {'num_atoms': num_atoms, 'positions': atom_position, 'ddec_charges': atom_ddec_charge}

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
                     bs == Chem.rdchem.BondStereo.STEREONONE, bs == Chem.rdchem.BondStereo.STEREOANY,
                     bs == Chem.rdchem.BondStereo.STEREOZ,
                     bs == Chem.rdchem.BondStereo.STEREOE,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


def num_atom_features():
    
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a, 0.0))


def num_bond_features():
    
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


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
        for i in range(start_line, end_line + 1):
            lines[i] = lines[i].decode('utf-8').replace("\n", "")
            lines[i] = [lines[i].split()[0], lines[i].split()[1], lines[i].split()[2]]
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


def prepare_dataset(datadir, dataset, subset=None, splits=None, copy=False):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces a fresh download of the dataset.

    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data.

    Notes
    -----
    TODO: Delete the splits argument?
    """

    
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    
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

    e4_data = {}
    train = []
    valid = []
    test = []
    data = []
    i = 0
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filepath in filenames:
            data_path = os.path.join(dirpath, filepath)
            print(data_path)
            if Chem.MolFromMolFile(data_path, removeHs=False) != None:
                data.append(process_sdf_e4(data_path))
                i = i + 1
                print('已完成:', i)

    data = convert(data)
    
    
    

    
    
    

    
    savedir = os.path.join(datadir, 'data.npz')
    np.savez_compressed(savedir, **data)
    print(savedir)
    print('successful')

    return datafiles






def main():
    datadir = '/home/suqun/score_test/MCL1'
    dataset = 'complex'
    prepare_dataset(datadir=datadir, dataset=dataset)


if __name__ == '__main__':
    main()
