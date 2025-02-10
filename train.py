from EGAT.egat_new2 import EGNN
import torch
from torch import nn, optim
import argparse
from data import arg_utils as qm9_utils, load_dataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import KFold
from utils.utils import initialize_datasets
from args import init_argparse
from dataset import ProcessedDataset
from utils.utils import _get_species
from utils.Myutils import *
import tqdm
from tqdm import tqdm
from datetime import datetime
import logging

logging.basicConfig(filename='resp_edge.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_time = datetime.now()
formatted_date = current_time.strftime("%m_%d_%H_%M")

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='RESP_noupdate', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
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
parser.add_argument('--patience', type=int, default=20, help="early stopping patience")
parser.add_argument('--use_edge_index', type=bool, default=False, help="early stopping patience")

arg = parser.parse_args()
arg.cuda = not arg.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if arg.cuda else "cpu")
dtype = torch.float32
args = init_argparse('qm9')




def back_true(atom_list, virtual_num, result):

    new_result = torch.cat([result[i * virtual_num:i * virtual_num + m] for i, m in enumerate(atom_list)])

    return new_result


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr': []}
    train_npred = []
    train_ntrue = []
    valid_npred = []
    valid_ntrue = []
    test_npred = []
    test_ntrue = []

    for i, data in tqdm(enumerate(loader)):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        _, num_bonds, _ = data['edge_index'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
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
            
        label = data[arg.property].view(batch_size * n_nodes, -1).to(device, dtype)

        pred = model(h0=nodes, edge=None, edge_index=edges, x=atom_positions, batch=batch)

        pred = back_true(data['num_atoms'], n_nodes, pred)
        label = back_true(data['num_atoms'], n_nodes, label)

        if partition == 'train':
            loss = loss_fn(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_npred.append(pred.detach())
            train_ntrue.append(label)


        elif partition == 'valid':
            with torch.no_grad():
                loss = loss_fn(pred, (label - meann) / mad)
                
                valid_npred.append(pred.detach())
                valid_ntrue.append(label)

        else:
            with torch.no_grad():
                loss = loss_fn(pred, (label - meann) / mad)
                test_npred.append(pred.detach())
                test_ntrue.append(label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

    if partition == 'train':

        train_true = torch.cat(train_ntrue).cpu().numpy()
        train_pred = torch.cat(train_npred).cpu().numpy()
        train_rmse = np.sqrt(mean_squared_error(mad * train_pred + meann, train_true))
        r2_train = r2_score(train_true, mad * train_pred + meann)
        rmse = train_rmse
        r2 = r2_train

    elif partition == 'valid':
        valid_true = torch.cat(valid_ntrue).cpu().numpy()
        valid_pred = torch.cat(valid_npred).cpu().numpy()
        valid_rmse = np.sqrt(mean_squared_error(mad * valid_pred + meann, valid_true))
        r2_valid = r2_score(valid_true, mad * valid_pred + meann)
        rmse = valid_rmse
        r2 = r2_valid

    else:
        test_true = torch.cat(test_ntrue).cpu().numpy()
        test_pred = torch.cat(test_npred).cpu().numpy()
        test_rmse = np.sqrt(mean_squared_error(mad * test_pred + meann, test_true))
        r2_test = r2_score(test_true, mad * test_pred + meann)
        rmse = test_rmse
        r2 = r2_test

    return res['loss'] / res['counter'], rmse, r2


if __name__ == "__main__":

    train_param = {'rmse': [], 'r2': []}
    test_param = {'rmse': [], 'r2': []}

    print(args.datadir)
    args, datasets = initialize_datasets(args, args.datadir, 'new_sdf')
    kf = KFold(n_splits=10)
    data = {}
    dataset_k = {}
    data = datasets['data']
    logging.info(f"train data length ：{len(data['positions'])}")
    count = 0
    for train_idx, test_idx in kf.split(data['positions']):
        count = count + 1
        global_rmse_best = 0.1
        loss_fn = nn.MSELoss()
        print(f"{count} begin training")

        model = EGNN(in_node_nf=68, in_edge_nf=0, hidden_nf=arg.nf, device=device, n_layers=arg.n_layers,
                     node_attr=arg.node_attr)

        optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arg.epochs)
        file_path = os.path.join(f'/home/suqun/model/EGAT/devir_result/ckpt/kfold/{arg.exp_name}' + str(count) + '.pth')
        stopper = EarlyStopping(mode='lower', patience=arg.patience, tolerance=arg.tolerance,
                                filename=file_path)

        dataset_k['train'] = {pro: data[pro][train_idx] for pro in data}
        dataset_k['test'] = {pro: data[pro][test_idx] for pro in data}

        all_species = _get_species(datasets)

        dataset_p = {split: ProcessedDataset(data, num_pts=-1, included_species=all_species) for split, data in
                     dataset_k.items()}
        dataloaders = load_dataset.retrieve_dataloaders(arg.batch_size, 5, dataset_k=dataset_p)

        
        meann, mad = qm9_utils.compute_mean_mad(dataloaders, arg.property)

        num_species = dataset_p['train'].num_species
        charge_scale = dataset_p['train'].max_charge

        print(f'mean: {meann} \t mad:, {mad}')

        for epoch in tqdm(range(arg.epochs)):
            a, train_rmse, r2_train = train(epoch, dataloaders['train'], partition='train')
            
            print(f'train_rmse:{train_rmse} \t r2_train:{r2_train}')
            train_param['rmse'].append(train_rmse)
            train_param['r2'].append(r2_train)
            if epoch % arg.test_interval == 0:

                test_loss, test_rmse, r2_test = train(epoch, dataloaders['test'], partition='test')

                test_param['rmse'].append(test_rmse)
                test_param['r2'].append(r2_test)

                print(f'test_rmse:{test_rmse} \t r2_test:{r2_test}')

                if test_rmse < global_rmse_best:
                    torch.save(model, f"/home/suqun/model/EGAT/devir_result/ckpt/best/{arg.exp_name}_{formatted_date}.pth")
                    global_rmse_best = test_rmse

                early_stop = stopper.step(test_rmse, model)
                if early_stop:
                    break

            xdata = pd.DataFrame({'train_rmse': train_param['rmse'], 'r2_train': train_param['r2'],
                                  'test_rmse': test_param['rmse'], 'r2_test': test_param['r2']})

            xdata.to_csv(f"./devir_result/data/{arg.exp_name}.csv", index=None)

