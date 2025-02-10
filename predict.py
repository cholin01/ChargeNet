from EGAT.egat_new2 import EGNN
import torch
import argparse
from data import arg_utils as qm9_utils, load_dataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from rdkit import Chem
from utils import initialize_datasets, main_utils
from args import init_argparse
from dataset import ProcessedDataset
from utils.Myutils import *
import tqdm
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='RESP', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=0.0005, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='ddec_charges', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=1, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
parser.add_argument('--patience', type=int, default=10, help="early stopping patience")
parser.add_argument('--use_edge_index', type=bool, default=False, help="early stopping patience")
parser.add_argument('--ckpt_path', type=str, default='/home/suqun/model/EGAT/end_result/ckpt/kfold/RESP1.pth')
parser.add_argument('--data_path', type=str, default='/home/suqun/score_test/MCL1/data.npz')
parser.add_argument('--meann', type=float, default='5.6592e-05')
parser.add_argument('--mad', type=float, default='0.1221')
parser.add_argument('--datadir', type=str, default='/home/suqun/score_test/MCL1')
parser.add_argument('--input_file', type=str, default='glide_active.sdf')
parser.add_argument('--output_file', type=str, default='MCL1_RESP.sdf')
parser.add_argument('--output_dir', type=str, default='MOL2_RESP')

arg = parser.parse_args()
arg.cuda = not arg.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if arg.cuda else "cpu")
dtype = torch.float32
args = init_argparse('qm9')

main_utils.makedir(arg.outf)
main_utils.makedir(arg.outf + "/" + arg.exp_name)


def bond_type_to_mol2(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return "1"
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return "2"
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return "3"
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return "ar"
    else:
        return "un"

def read_sdf_file(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    return mols


def write_sdf_with_atom_features(mol, row, filename):
    

    

    num_atoms = mol.GetNumAtoms()

    for i in range(num_atoms):
        mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(row[i]))
        

    
    writer = Chem.SDWriter(os.path.join(arg.datadir, 'complex', filename.split("_")[0], f"{filename}_RESP.sdf"))
    writer.write(mol)
    writer.close()


def create_mol2_with_charge(mol, output_file, row):
    
    mol2_content = []
    mol2_content.append("@<TRIPOS>MOLECULE\n")
    mol2_content.append("Molecule\n")
    mol2_content.append(f" {mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0\n")
    mol2_content.append("SMALL\n")
    mol2_content.append("USER_CHARGES\n\n")

    
    mol2_content.append("@<TRIPOS>ATOM\n")
    for atom in mol.GetAtoms():
        idx = atom.GetIdx() + 1
        pos = mol.GetConformer().GetAtomPosition(idx - 1)
        charge = float(row[idx - 1])
        mol2_content.append(
            f"{idx:>7} {atom.GetSymbol():<2} {pos.x:>10.4f} {pos.y:>10.4f} {pos.z:>10.4f} {atom.GetSymbol():<2} 1 <0> {charge:>10.4f}\n")

    
    mol2_content.append("@<TRIPOS>BOND\n")
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx() + 1
        idx2 = bond.GetEndAtomIdx() + 1
        bond_type = bond_type_to_mol2(bond)
        mol2_content.append(f"{bond.GetIdx() + 1:>6} {idx1:>4} {idx2:>4} {bond_type}\n")

    
    with open(output_file, 'w') as file:
        file.writelines(mol2_content)


def write_mol2_with_atom_features(input_file, output_file, row):
    

    with open(input_file, 'r') as f:
        mol2_content = f.readlines()

        
    atom_start_idx = mol2_content.index('@<TRIPOS>ATOM\n') + 1

    atom_end_idx = mol2_content.index('@<TRIPOS>BOND\n') - 1

    length = atom_end_idx - atom_start_idx

    row_charge = row[:length]

    
    for idx, charge in zip(range(atom_start_idx, atom_end_idx), row_charge):
        
        charge_start_pos = mol2_content[idx].rfind(" ") + 1

        
        mol2_content[idx] = mol2_content[idx][:charge_start_pos] + f"{charge:.4f}\n"

    
    with open(output_file, 'w') as f:
        f.writelines(mol2_content)


def charge_predict(loader, new_mol=False):
    mol_charge = {}
    predictions = []
    error_list = []
    labels = []
    atomic_num = []
    file_names = []

    for i, data in tqdm(enumerate(loader)):
        model.eval()

        with torch.no_grad():

            batch_size, n_nodes, _ = data['positions'].size()
            _, num_bonds, _ = data['edge_index'].size()
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            try:
                label = data['ddec_charges'].to(device, dtype)
            except:
                print('no label')
            mol_atom_prop = data['mol_atom_prop'].to(device, dtype)
            mol_bond_prop = data['mol_bond_prop'].to(device, dtype)

            nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
            nodes = torch.cat([mol_atom_prop, nodes], dim=2)
            nodes = nodes.view(batch_size * n_nodes, -1).to(device)

            batch = [torch.tensor(i) for i in range(batch_size) for j in range(n_nodes)]
            batch = torch.stack(batch).to(device)

            if arg.use_edge_index:
                edge_index = data['edge_index'].view(batch_size * num_bonds, -1).T
                edges = qm9_utils.handle_edge_index(edge_index, batch_size, num_bonds, n_nodes, device)
            else:
                edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)

            mol_index = data['filename']

            

            pred = model(h0=nodes, edge=None, edge_index=edges, x=atom_positions, batch=batch)

            predict1 = arg.mad * pred.detach().cpu() + arg.meann
            predict = predict1.view(batch_size, n_nodes, -1)
            predictions.extend(predict1.numpy())
            atomic_num.extend(charges.view(batch_size * n_nodes).detach().cpu().numpy())
            try:
                labels.extend(label.view(batch_size * n_nodes).detach().cpu().numpy())
            except:
                print('there is no labels')

                print(predict1.size())
                print(charges.size())
                

            for i in range(len(mol_index)):
                mol_charge[mol_index[i]] = predict[i].T

    result = {key: (pad_sequence(val, batch_first=True)) for key, val in mol_charge.items()}
    result = {key: np.array(val.squeeze(0)) for key, val in result.items()}
    df = pd.DataFrame.from_dict(result, orient='index')
    
    

    if new_mol:

        
        
        
        
        
        
        
        
        
        
        


        file_path = os.path.join(arg.datadir, arg.input_file)
        target_path = os.path.join(arg.datadir, arg.output_file)
        writer = Chem.SDWriter(target_path)
        mol_list = Chem.SDMolSupplier(file_path, removeHs=False)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        mol_data = {index: row for index, row in df.iterrows()}

        
        for mol in tqdm(mol_list):
            mol_name = mol.GetProp('_Name')

            
            if mol_name in mol_data:
                row = mol_data[mol_name]
                num_atoms = mol.GetNumAtoms()

                
                for i in range(min(num_atoms, len(row))):
                    rounded_value = np.round(row[i], 5)
                    mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(rounded_value))

                writer.write(mol)
            else:
                print(f'Bad mol file: {mol_name}')

        writer.close()

    print(len(predictions))
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1)
    labels = np.array(labels)
    labels = labels.reshape(-1)

    data = {'atomic_number': atomic_num, 'predictions': predictions, 'labels': labels}

    df_2 = pd.DataFrame(data)

    df_2 = df_2[df_2['atomic_number'] != 0]

    csv_name = '/home/suqun/model/EGAT/submit_data/extra_data/resp.csv'
    df_2.to_csv(csv_name, index=False)

    test_rmse = np.sqrt(mean_squared_error(predictions, labels))
    r2_test = r2_score(labels, predictions)

    print(f'test_rmse = {test_rmse}')
    print(f'test_r2 = {r2_test}')

    return 0


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    train_param = {'rmse': [], 'r2': []}
    valid_param = {'rmse': [], 'r2': []}
    test_param = {'rmse': [], 'r2': []}

    print(arg.datadir)
    args, datasets = initialize_datasets(args, arg.datadir, 'complex', arg.data_path)
    data = {}
    dataset_k = {}
    data = datasets['data']
    count = 0

    model = EGNN(in_node_nf=68, in_edge_nf=0, hidden_nf=arg.nf, device=device, n_layers=arg.n_layers,
                 node_attr=arg.node_attr)

    
    model_state_dict = torch.load(arg.ckpt_path, map_location=device)

    model.load_state_dict(model_state_dict['model_state_dict'])

    all_species = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

    all_species = torch.tensor(all_species)

    dataset_p = {split: ProcessedDataset(data, num_pts=-1, included_species=all_species) for split, data in
                 datasets.items()}

    num_species = dataset_p['data'].num_species

    charge_scale = dataset_p['data'].max_charge
    print(charge_scale)

    
    assert (len(set(tuple(data.included_species.tolist()) for data in dataset_p.values())) ==
            1), 'All datasets must have same included_species! {}'.format(
        {key: data.included_species for key, data in dataset_p.items()})

    print("arg.num_workers: {}".format(arg.num_workers))
    dataloaders = load_dataset.retrieve_dataloaders(arg.batch_size, arg.num_workers, dataset_k=dataset_p)

    charge_predict(dataloaders['data'])
    
