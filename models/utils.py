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



