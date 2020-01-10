import torch
from torch import nn
from models.utils import SimplerLinear

def get_combinator(type_id, in_size = None):
    types = {c.__name__:c for c in (Add, Mul, Average)}
    if type_id in types:
        return types[type_id]()
    return Interpolation(type_id, in_size)

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rightwards, embeddings, existences):
        if rightwards is None:
            lw_emb = embeddings[:,  1:]
            rw_emb = embeddings[:, :-1]
            lw_ext = existences[:,  1:]
            rw_ext = existences[:, :-1]
        else:
            lw_ext = (existences & ~rightwards)[:,  1:]
            rw_ext = (existences &  rightwards)[:, :-1]
            lw_relay = lw_ext & ~rw_ext
            rw_relay = ~lw_ext & rw_ext

            right = rightwards.type(embeddings.dtype)
            # right.unsqueeze_(-1)
            lw_emb = embeddings[:,  1:] * (1 - right)[:,  1:]
            rw_emb = embeddings[:, :-1] *      right [:, :-1]

        add_emb = lw_emb + rw_emb
        cmp_emb = self.compose(lw_emb, rw_emb)
        new_jnt = lw_ext & rw_ext
        new_ext = lw_ext | rw_ext
        if cmp_emb is None:
            new_emb = add_emb
        else:
            new_emb = torch.where(new_jnt, cmp_emb, add_emb)

        if rightwards is None:
            return new_ext, new_emb
        return new_ext, new_jnt, lw_relay, rw_relay, new_emb

    def compose(self, lw_emb, rw_emb):
        return None # Default by add

class Mul(Add):
    def compose(self, lw_emb, rw_emb):
        return lw_emb * rw_emb

class Average(Add):
    def compose(self, lw_emb, rw_emb):
        return lw_emb * rw_emb / 2

# class Max(Add):
# class Cos(Add):

class Interpolation(Add):
    def __init__(self, type_id, in_size, out_size = None, bias = True):
        super().__init__()
        if out_size is None:
            out_size = in_size
        else:
            raise NotImplementedError()

        self._activation = activation = nn.Sigmoid()
        if type_id[0] == 'N':
            assert bias, f'invalid {type_id} without a bias parameter'
            if type_id == 'NV':
                itp_ = SimplerLinear(in_size, weight = False)
            elif type_id == 'NS':
                itp_ = SimplerLinear(1, weight = False)
            # extra_repr = type_id + ': ' + str(itp_)
            self._itp = itp_
            def _compose(lw, rw):
                itp = activation(itp_(1))
                return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'C' and type_id[2] in '12':
            if type_id == 'CV2':
                itp_l = nn.Linear(in_size, out_size, bias = False)
                itp_r = nn.Linear(in_size, out_size, bias = bias)
            elif type_id == 'CS2':
                itp_l = nn.Linear(in_size, 1, bias = False)
                itp_r = nn.Linear(in_size, 1, bias = bias)
            elif type_id == 'CV1':
                itp_l = SimplerLinear(in_size, bias = False)
                itp_r = SimplerLinear(in_size, bias = bias)
            elif type_id == 'CS1':
                itp_l = SimplerLinear(1, bias = False)
                itp_r = SimplerLinear(1, bias = bias)
            # extra_repr = f'{type_id}: {itp_l} & {itp_r}'
            self._itp_l = itp_l # pytorch direct register? yes
            self._itp_r = itp_r # even not uses
            def _compose(lw, rw):
                # lw =  # Either Vector
                # rw =  # Or Scalar to *
                itp = activation(itp_l(lw) + itp_r(rw)) # Concatenate
                return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'B':
            if type_id == 'BV':
                itp_ = nn.Bilinear(in_size, in_size, out_size, bias = bias)
            elif type_id == 'BS':
                itp_ = nn.Bilinear(in_size, in_size, 1, bias = bias)
            # extra_repr = type_id + ': ' + str(itp_)
            self._itp = itp_
            def _compose(lw, rw):
                itp = itp_(lw, rw)
                itp = activation(itp)
                return (1 - itp) * lw + itp * rw

        # self._extra_repr = extra_repr
        self._compose = _compose

    def compose(self, lw_emb, rw_emb):
        return self._compose(lw_emb, rw_emb)

    # def extra_repr(self):
    #     return self._extra_repr