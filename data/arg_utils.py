import torch

def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def get_adj_matrix_edit(adj_matrix, n_nodes, batch_size, device):


    n = len(adj_matrix)
    row_all = []
    col_all = []
    for i in range(n):
        

        
        print(adj_matrix[i])
        rows, cols = torch.where(adj_matrix[i] == 1)

        row_all.append(rows + i * n_nodes)
        col_all.append(cols + i * n_nodes)
        
        

        
    row_all = torch.cat(row_all, dim=0).to(device)
    col_all = torch.cat(col_all, dim=0).to(device)

    keep_indices = []
    for i in range(len(row_all)):
        if row_all[i] != col_all[i]:
            keep_indices.append(i)

    row_all = row_all[keep_indices]
    col_all = col_all[keep_indices]

    edges = [row_all, col_all]

    return edges

def handle_edge_index(edge_index, batch_size, num_bonds, n_nodes, device):

    for i in range(edge_index.size()[1]):
        edge_index[0][i] = edge_index[0][i] + n_nodes * (i // num_bonds)
        edge_index[1][i] = edge_index[1][i] + n_nodes * (i // num_bonds)
    edges = edge_index.type(torch.long).T

    
    all_same = (edges == edges[:, 0][:, None]).all(axis=1)

    
    edges = edges[~all_same]

    edges = torch.cat(
        (edges.to(device), (torch.tensor([[batch_size * n_nodes - 1, batch_size * n_nodes - 1]])).to(device)), dim=0).T

    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars
