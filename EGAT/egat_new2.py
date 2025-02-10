
import math
import e3nn
from e3nn import o3
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, scatter_mean
from graph_norm import EquivariantGraphNorm
from drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath



class EGAT(nn.Module):
    def __init__(self, dim_in, hidden_dim, edge_in, edge_out, num_head=4, drop_rate=0.15, gated=False):
        super().__init__()
        self.gated = gated
        self.irrep_out = o3.Irreps('1x0e')
        self.edge_dim = edge_in
        self.num_head = num_head  
        self.dh = hidden_dim // num_head  
        self.hidden_dim = hidden_dim  
        self.src_encode = nn.Linear(dim_in, hidden_dim)
        self.dst_encode = nn.Linear(dim_in, hidden_dim)
        self.value_encode = nn.Linear(dim_in, hidden_dim)
        self.edge_layer = nn.Sequential(
            nn.Linear(edge_in + 1, hidden_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=drop_rate))
        self.update_edge = nn.Sequential(
            nn.Linear(hidden_dim, edge_out),
            nn.Dropout(p=drop_rate))
        self.gate_layer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Dropout(p=drop_rate))
        
        self.edge_gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        self.coord_update = nn.Sequential(
            nn.Linear(self.dh, self.dh // 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(self.dh // 2, 1)
        )
        self.head_layer = nn.Linear(num_head, 1, bias=False)

        self.graph_norm1 = GraphNorm(hidden_dim)
        self.graph_norm2 = GraphNorm(hidden_dim)
        
        
        
        self.fix_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def coord_function(self, edge_result, pos, edge_index):
        i, j = edge_index
        radial = pos[i] - pos[j]
        radial = radial / (torch.norm(radial, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6)
        radial = radial * self.head_layer(self.coord_update(edge_result).squeeze(dim=2))
        radial = scatter(radial, index=i, reduce='sum', dim=0)
        pos += radial
        return pos

    def edge_function(self, edge, edge_index, coords):

        if edge is not None:
            edge = torch.cat([edge, torch.norm(coords[edge_index[0]] - coords[edge_index[1]], p=2, dim=-1, keepdim=True)
                              * 0.1], dim=-1)
        else:
            edge = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], p=2, dim=-1, keepdim=True) * 0.1

        edge = self.edge_layer(edge)
        
        if self.gated:
            edge = edge * self.edge_gate(edge)
            
        return edge

    def node_function(self, node, edge_result, edge_weight, value, edge_index, batch):

        node_new = self.update_node(
            scatter(edge_weight * value[edge_index[1]].view((-1, self.num_head, self.dh)), index=edge_index[0],
                    reduce='sum',
                    dim=0).view((-1, self.hidden_dim)))
        edge_new = self.update_edge(edge_result.view((-1, self.hidden_dim)))
        g = torch.sigmoid(self.gate_layer(torch.cat([node_new, node, node_new - node], dim=-1)))
        node_new = self.graph_norm1(g * node_new + node, batch)
        node_new = self.graph_norm2(g * self.fix_node(node_new) + node_new, batch)

        return node_new, edge_new


    def forward(self, node, edge, edge_index, coords, batch, update_pos=True):
        
        src = self.src_encode(node)
        dst = self.dst_encode(node)
        value = self.value_encode(node)

        
        edge = self.edge_function(edge, edge_index, coords)

        edge_key = src[edge_index[1]] * edge
        edge_result = ((dst[edge_index[0]] * edge_key) / math.sqrt(self.dh)).view((-1, self.num_head, self.dh))
        edge_weight = softmax(torch.norm(edge_result, p=1, dim=2), index=edge_index[0]).unsqueeze(dim=-1)

        
        node_new, edge_new = self.node_function(node, edge_result, edge_weight, value, edge_index, batch)

        
        

        return node_new, edge_new, coords        

   


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=8, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr

        for i in range(0, n_layers):
            self.add_module("gat_%d" % i, EGAT(self.hidden_nf, self.hidden_nf, edge_in=in_edge_nf, edge_out=128))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, edge, edge_index, x, batch):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            
            h, _, _ = self._modules["gat_%d" % i](h, edge, edge_index, x, batch)

        h = self.node_dec(h)
        pred = self.graph_dec(h)

        return pred



