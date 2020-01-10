import torch
class PCA:
    def __init__(self, emb, k = 9):
        emb_mean = torch.mean(emb, dim = 0)
        emb_shifted = emb - emb_mean
        emb_cov = torch.matmul(emb_shifted.T, emb_shifted)
        val, vec = torch.eig(emb_cov, True)
        _, idx = val[:, 0].topk(k) # ignore value & image part
        self._bases = vec[:, idx] # both tensorflow and torch use dim = 1

    def __call__(self, emb):
        m_ = (emb * emb).mean(-1, keepdim = True)
        pc = torch.matmul(emb, self._bases)
        return torch.cat([m_, pc], -1)

def fraction(cnt_n, cnt_d, dtype = torch.float32):
    return cnt_n.sum().type(dtype) / cnt_d.sum().type(dtype)

import math
from torch.nn import Module, Parameter, init
class SimplerLinear(Module):
    __constants__ = ['bias', 'in_features']
    def __init__(self, in_features, weight = True, bias = True):
        assert weight or bias
        super(SimplerLinear, self).__init__()
        self.in_features = in_features
        if weight:
            self.weight = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(5)
        if self.weight is not None:
            init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is not None:
            input = input * self.weight
        if self.bias is not None:
            input = input + self.bias
        return input

    def extra_repr(self):
        return 'in_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )

class GaussianCodebook(Module):
    __constants__ = ['bias', 'codebook' 'io_features']
    def __init__(self, in_dim, num_codes, prefix_dims = 0):
        super(GaussianCodebook, self).__init__()

        self.io_features = in_dim, num_codes
        size = [1 for _ in range(prefix_dims)]
        size += [in_dim, num_codes]
        self.codebook = Parameter(torch.Tensor(*size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.codebook, a = math.sqrt(5))

    def forward(self, x):
        diff = (x.unsqueeze(dim = 3) - self.codebook) ** 2
        return diff.mean(dim = -2) # [?...?, c]

    def repulsion(self, coeff):
        in_dim, num_codes = self.io_features
        code = self.codebook.view(1, in_dim, num_codes)
        _code = code.transpose(0, 2)
        ccd = (code - _code) ** 2 # [c, h, c]
        ccd = ccd.mean(dim = 1) # [c, c]
        ccg = - ccd * coeff
        ccg.exp_() # repulsion
        return ccg.sum() # minimize

    def distance(self, logtis, oracle):
        # [b,s,c][b,s]
        oracle = oracle.unsqueeze(dim = 2)
        distance = logtis.gather(dim = 2, index = oracle)
        distance.squeeze_(dim = 2)
        return distance

    def extra_repr(self):
        in_features, out_features = self.io_features
        return 'in_dim={}, num_codes={}'.format(
            in_features, out_features
        )

def squeeze_left(hidden, existence_or_start, as_existence = False, offset = None, out_len = None, get_rid_of_last_k = 0):
    seq_idx = torch.cumsum(existence_or_start, 1)

    if offset is not None:
        seq_idx += offset[:, None]

    # 0 used as a dump
    if get_rid_of_last_k:
        max_idx = seq_idx[:, -1:] - get_rid_of_last_k
        seq_idx[seq_idx > max_idx] = 0
        
    if as_existence:
        seq_idx *= existence_or_start
        
    max_len = seq_idx.max() + 1

    # from hidden redirected by seq_idx to zeros, throw away 1:
    hidden_shape = list(hidden.shape)
    if out_len is None:
        hidden_shape[1] += 1
        truncate = max_len
    elif hidden_shape[1] < out_len:
        hidden_shape[1] = out_len + 1
        truncate = None
    else:
        hidden_shape[1] += 1
        truncate = out_len + 1
        
    if len(hidden_shape) == 3:
        seq_idx.unsqueeze_(dim = -1)
        cumu_shape = hidden_shape[:-1] + [1]
    else:
        cumu_shape = hidden_shape

    base = torch.zeros(*hidden_shape, device = hidden.device, dtype = hidden.dtype)
    cumu = torch.zeros(*cumu_shape, device = hidden.device, dtype = seq_idx.dtype)
    if as_existence:
        base = base.scatter(1, seq_idx.expand_as(hidden), hidden)
    else:
        base = base.scatter_add(1, seq_idx.expand_as(hidden), hidden)
    cumu.scatter_add_(1, seq_idx, torch.ones_like(seq_idx))

    # 0 used as a dump
    if truncate is None:
        base = base[:, 1:]
        cumu = cumu[:, 1:]
    else:
        base = base[:, 1:truncate]
        cumu = cumu[:, 1:truncate]
    if as_existence:
        return base, cumu > 0
    return base, cumu