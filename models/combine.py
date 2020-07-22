import torch
from torch import nn
from models.utils import SimplerLinear, condense_helper, condense_left, release_left
from models.backend import activation_type
from utils.types import BaseType
from utils.str_ops import is_numeric

E_COMBINE = 'CV2 CV1 CS2 CS1 Add Mul Average NV NS BV BS'.split()
def valid_transform(x): # CT.Tanh.3
    if x.startswith('CT') or x.startswith('BT'):
        segs = x.split('-')
        if len(segs) > 3:
            return False
        valid = activation_type.validate(segs[1])
        if len(segs) == 2:
            return valid
        return valid and is_numeric.fullmatch(segs[2])
    return False
combine_type = BaseType(0, as_index = True, default_set = E_COMBINE, validator = valid_transform)

def get_combinator(type_id, in_size = None):
    types = {c.__name__:c for c in (Add, Mul, Average)}
    if type_id in types:
        return types[type_id]()
    return Interpolation(type_id, in_size)

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt):
        if rightwards_or_lhs.shape == embeddings_or_rhs.shape:
            self.disco_forward(rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt)
        else:
            self.conti_forward(rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt)

    def disco_forward(self, lhs, rhs, phy_jnt):
        cmp_emb = self.compose(lhs, rhs, phy_jnt)
        return torch.where(phy_jnt, cmp_emb, lhs)
    
    def conti_forward(self, rightwards, embeddings, existences):
        if existences is None:
            assert rightwards is None
            lw_emb = embeddings[1:]
            rw_emb = embeddings[:-1]
            return self.compose(lw_emb, rw_emb, None)
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

        new_jnt = lw_ext & rw_ext
        new_ext = lw_ext | rw_ext
        add_emb = lw_emb + rw_emb
        cmp_emb = self.compose(lw_emb, rw_emb, new_jnt)
        if cmp_emb is None:
            new_emb = add_emb
        else:
            new_emb = torch.where(new_jnt, cmp_emb, add_emb)

        if rightwards is None:
            return new_ext, new_emb
        return new_ext, new_jnt, lw_relay, rw_relay, new_emb

    def compose(self, lw_emb, rw_emb, is_jnt):
        return None # Default by add

class Mul(Add):
    def compose(self, lw_emb, rw_emb, is_jnt):
        return lw_emb * rw_emb

class Average(Add):
    def compose(self, lw_emb, rw_emb, is_jnt):
        return lw_emb * rw_emb / 2

# class Max(Add):
# class Cos(Add):

class Interpolation(Add):
    def __init__(self, type_id, in_size, out_size = None, bias = True):
        super().__init__()
        use_condenser = False
        if out_size is None:
            out_size = in_size
        else:
            raise NotImplementedError()

        if type_id in E_COMBINE:
            activation = nn.Sigmoid()
        else:
            segs = type_id.split('-')
            activation = activation_type[segs[1]]()
            scale = float(segs[2]) if len(segs) == 3 else None
            print(type_id, activation, scale)
        self._activation = activation
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
        elif type_id[0] == 'C':
            if type_id =='CV2' or type_id.startswith('CT-'):
                # use_condenser = True # TESTED! 100sps SLOWER
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
            if type_id == 'CT':
                def _compose(lw, rw):
                    tsf = itp_l(lw) + itp_r(rw) # Concatenate
                    if scale is not None:
                        tsf = activation(tsf) * scale
                    else:
                        tsf = activation(tsf)
                    return tsf
            else:
                def _compose(lw, rw):
                    # lw =  # Either Vector
                    # rw =  # Or Scalar to *
                    itp = activation(itp_l(lw) + itp_r(rw)) # Concatenate
                    return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'B':
            if type_id == 'BV' or type_id.startswith('BT-'):
                use_condenser = True
                itp_ = nn.Bilinear(in_size, in_size, out_size, bias = bias)
            elif type_id == 'BS':
                itp_ = nn.Bilinear(in_size, in_size, 1, bias = bias)
            # extra_repr = type_id + ': ' + str(itp_)
            self._itp = itp_
            if type_id == 'BT':
                def _compose(lw, rw):
                    if type_id != 'BS':
                        lw = lw.contiguous()
                        rw = rw.contiguous()
                    if scale is not None:
                        tsf = activation(itp_(lw, rw)) * scale
                    else:
                        tsf = activation(itp_(lw, rw))
                    return tsf
            else:
                def _compose(lw, rw):
                    if type_id != 'BS':
                        lw = lw.contiguous()
                        rw = rw.contiguous()
                    itp = itp_(lw, rw)
                    itp = activation(itp)
                    return (1 - itp) * lw + itp * rw

        # self._extra_repr = extra_repr
        self._use_condenser = use_condenser
        self._compose = _compose

    def compose(self, lw_emb, rw_emb, is_jnt):
        if self._use_condenser and is_jnt is not None and is_jnt.shape[1] > 1:
            helper = condense_helper(is_jnt.squeeze(dim = 2), as_existence = True)
            cds_lw, seq_idx = condense_left(lw_emb, helper, get_indice = True)
            cds_rw          = condense_left(rw_emb, helper)
            cds_cmb = self._compose(cds_lw, cds_rw)
            return release_left(cds_cmb, seq_idx)
        return self._compose(lw_emb, rw_emb)

    # def extra_repr(self):
    #     return self._extra_repr