import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn.o3 import Irreps



class EquivariantGraphNorm(nn.Module):
    '''Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    '''

    def __init__(self, irreps, eps=1e-5, affine=True, reduce='mean', normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        self.mean_shift = nn.Parameter(torch.ones(num_scalar))
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"


    
    def forward(self, node_input, batch, **kwargs):
        '''evaluate
        Parameters
        ----------
        node_input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        '''
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for mul, ir in self.irreps:  
            d = ir.dim
            
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d

            
            field = field.reshape(-1, mul, d)

            
            if ir.l == 0 and ir.p == 1:
                
                field_mean = global_mean_pool(field, batch).reshape(-1, mul, 1)  
                
                mean_shift = self.mean_shift[i_mean_shift : (i_mean_shift + mul)]
                mean_shift = mean_shift.reshape(1, mul, 1)
                field = field - field_mean[batch] * mean_shift

            
            
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            
            if self.reduce == 'mean':
                field_norm = global_mean_pool(field_norm, batch)  
            elif self.reduce == 'max':
                field_norm = global_max_pool(field_norm, batch)  
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            
            field_norm = (field_norm + self.eps).pow(-0.5)  

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  
                iw += mul
                field_norm = field_norm * weight  

            field = field * field_norm[batch].reshape(-1, mul, 1)  

            if self.affine and d == 1 and ir.p == 1:  
                bias = self.affine_bias[ib: ib + mul]  
                ib += mul
                field += bias.reshape(mul, 1)  

            
            fields.append(field.reshape(-1, mul * d))  

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  
        return output


class EquivariantGraphNormV2(nn.Module):
    
    def __init__(self, irreps, eps=1e-5, affine=True, reduce='mean', normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        mean_shift = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                mean_shift.append(torch.ones(1, mul, 1))
            else:
                mean_shift.append(torch.zeros(1, mul, 1))
        mean_shift = torch.cat(mean_shift, dim=1)
        self.mean_shift = nn.Parameter(mean_shift)
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"


    
    def forward(self, node_input, batch, **kwargs):
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        node_input_mean = global_mean_pool(node_input, batch)
        for mul, ir in self.irreps:
            
            print(mul, ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            field = field.reshape(-1, mul, d) 
            
            
            field_mean = node_input_mean.narrow(1, ix, mul * d)
            field_mean = field_mean.reshape(-1, mul, d)
            ix += mul * d
            
            mean_shift = self.mean_shift.narrow(1, i_mean_shift, mul)
            field = field - field_mean[batch] * mean_shift
            i_mean_shift += mul
            
            
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  
            
            if self.reduce == 'mean':
                field_norm = global_mean_pool(field_norm, batch)  
            elif self.reduce == 'max':
                field_norm = global_max_pool(field_norm, batch)  
            
            
            field_norm = (field_norm + self.eps).pow(-0.5)  

            if self.affine:
                weight = self.affine_weight.narrow(1, iw, mul)  
                iw += mul
                field_norm = field_norm * weight  

            field = field * field_norm[batch].reshape(-1, mul, 1)  

            if self.affine and d == 1 and ir.p == 1:  
                bias = self.affine_bias.narrow(1, ib, mul)  
                ib += mul
                field = field + bias.reshape(1, mul, 1)  

            fields.append(field.reshape(-1, mul * d))  

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  
        return output

