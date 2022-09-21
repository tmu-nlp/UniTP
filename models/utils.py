import torch
class PCA:
    def __init__(self, emb, k = 9):
        emb = emb.detach()
        emb_mean = torch.mean(emb, dim = 0)
        emb_shifted = emb - emb_mean
        emb_cov = torch.matmul(emb_shifted.T, emb_shifted)
        val, vec = torch.linalg.eig(emb_cov)
        _, idx = val.real.topk(k) # ignore value & image part
        self._bases = vec.real[:, idx] # both tensorflow and torch use dim = 1

    def __call__(self, emb):
        m_ = (emb * emb).mean(-1, keepdim = True)
        pc = torch.matmul(emb, self._bases)
        return torch.cat([m_, pc], -1)

def mean_stdev(weights, dim = -1):
    wm = weights.mean(dim = dim, keepdim = True)
    sd = (weights - wm) ** 2
    sd = sd.sum(dim = dim, keepdim = True) / (weights.shape[dim] - 1)
    sd = torch.sqrt(sd)
    return torch.cat([wm, sd], dim = dim)

def fraction(cnt_n, cnt_d = None, dtype = torch.float32):
    if cnt_d is None:
        return cnt_n.sum().type(dtype) / cnt_n.nelement()
    return cnt_n.sum().type(dtype) / cnt_d.sum().type(dtype)

def hinge_score(hinge_logits, inplace):
    if inplace:
        hinge_scores = hinge_logits
        hinge_scores += 1
    else:
        hinge_scores = hinge_logits + 1
        
    hinge_scores /= 2
    hinge_scores[hinge_scores < 0] = 0
    hinge_scores[hinge_scores > 1] = 1

    if not inplace:
        return hinge_scores

def distance(lhs, rhs):
    diff = lhs - rhs
    diff = (diff ** 2).sum(2)
    return diff.sqrt() / lhs.shape[1]

def cosine(lhs, rhs):
    lr = (lhs * rhs).sum(2)
    l2 = (lhs * lhs).sum(2)
    r2 = (rhs * rhs).sum(2)
    return lr / (l2 * r2)

def reduce_matrix(lhs, rhs, fn): # [num, dim]
    return fn(lhs.unsqueeze(1).detach(), rhs.unsqueeze(0).detach()).cpu().numpy()

def get_multilingual(layer):
    corps = list(layer.keys())
    for eid, lhs in enumerate(corps):
        for rhs in corps[eid:]:
            d = reduce_matrix(layer[lhs].weight, layer[rhs].weight, distance)
            c = reduce_matrix(layer[lhs].weight, layer[rhs].weight, cosine)
            yield lhs, rhs, d, c


import math
from torch.nn import Module, Parameter, init, Dropout
from torch.nn import Linear, Softmax, Softmin, ModuleDict
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
        bound = 1 / math.sqrt(self.in_features)
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

class BiaffineAttention(Module):
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    """
    __constants__ = ['weight', 'bias']
    def __init__(self, lhs_features, rhs_features, bias = False):
        super(BiaffineAttention, self).__init__()
        self.lhs_features = lhs_features
        self.rhs_features = rhs_features
        self.weight = Linear(lhs_features, rhs_features, bias = False)
        if bias:
            self.bias = Bias(1)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, lhs, rhs):
        hidden = self.weight(lhs)
        hidden = hidden.bmm(rhs)
        if self.bias is not None:
            hidden = hidden + self.bias()
        return hidden

    def reset_parameters(self):
        self.weight.reset_parameters()
        if self.bias is not None:
            self.bias.reset_parameters()

class LinearBinary(Module):
    __constants__ = ['weight_in', 'weight_hidden']
    def __init__(self, in_features, hidden_features = 1, in_act = None):
        super(LinearBinary, self).__init__()
        self.in_features     = in_features
        self.hidden_features = hidden_features
        self.weight_in = Linear(in_features, hidden_features)
        if hidden_features > 1:
            assert in_act is not None, 'Hidden layer need an activation'
            self.in_act = in_act()
            self.weight_hidden = Linear(hidden_features, 1)
        else:
            self.register_parameter('weight_hidden', None)
        self.reset_parameters()

    def forward(self, embed, drop_out = None):
        hidden = self.weight_in(embed)
        if self.weight_hidden is not None:
            if callable(drop_out):
                hidden = drop_out(hidden)
            hidden = self.weight_hidden(self.in_act(hidden))
        return hidden.squeeze(dim = -1)

    def reset_parameters(self):
        self.weight_in.reset_parameters()
        if self.weight_hidden is not None:
            self.weight_hidden.reset_parameters()

class LinearM2(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act, drop_out):
        super().__init__()
        self._hidden_dim = hidden_dim
        if hidden_dim:
            self._l0 = Linear(in_dim, hidden_dim)
            self._l1 = Linear(hidden_dim, out_dim)
            self._act = act()
            self._dp = Dropout(drop_out)
        else:
            self._l0

    def forward(self, emb):
        if self._hidden_dim:
            hidden = self._l0(emb)
            hidden = self._dp(hidden)
            hidden = self._act(hidden)
            return self._l1(hidden)
        return self._l0(emb)

class Bias(Module):
    __constants__ = ['bias']
    def __init__(self, *bias_shape):
        super(Bias, self).__init__()
        self.bias_shape = bias_shape
        self.bias = Parameter(torch.empty(bias_shape))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(sum(self.bias_shape))
        init.uniform_(self.bias, -bound, bound)

    def forward(self):
        return self.bias

    def extra_repr(self):
        return 'bias_shape={}'.format(self.in_features)

def get_logit_layer(logit_type):
    if logit_type in ('affine', 'linear'):
        Net = lambda i_size, o_size: Linear(i_size, o_size, bias = logit_type == 'affine')
        argmax = True
        score_fn = Softmax
    elif logit_type.startswith('codebook'):
        argmax = False
        if '|' in logit_type:
            bar = logit_type.index('|') + 1
            repulsion = float(logit_type[bar:])
        else:
            repulsion = 0
        score_fn = Softmin
        Net = lambda i_size, o_size: GaussianCodebook(i_size, o_size, coeff = repulsion)
    def net_fn(i_size, o_size):
        if isinstance(o_size, int):
            return Net(i_size, o_size)
        return ModuleDict({k: Net(i_size, v) for k,v in o_size.items()})
    return net_fn, argmax, score_fn

class GaussianCodebook(Module):
    __constants__ = ['codebook' 'io_features']
    def __init__(self, in_dim, num_codes, prefix_dims = 0, coeff = 0):
        super(GaussianCodebook, self).__init__()

        self.io_features = in_dim, num_codes
        self._coeff = coeff
        size = [1 for _ in range(prefix_dims)]
        size += [in_dim, num_codes]
        self.codebook = Parameter(torch.Tensor(*size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.codebook, a = math.sqrt(5))

    def forward(self, x):
        diff = (x.unsqueeze(dim = 3) - self.codebook) ** 2
        return diff.mean(dim = -2) # [?...?, c]

    def repulsion(self, coeff = 0):
        if coeff <= 0:
            if self._coeff <= 0:
                return 0
            coeff = self._coeff
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

def bos_mask(seq_len, offset):
    mask = torch.arange(seq_len, device = offset.device)[None]
    return mask < offset[:, None]

def eos_mask(seq_len, length):
    mask = torch.arange(seq_len, device = length.device)[None]
    return mask >= length[:, None]

def condense_splitter(right_layer, joint_layer, existence):
    batch_size, old_batch_len = existence.shape
    agree_orient = right_layer[:, :-1] & existence[:, :-1] # lhs2r
    agree_orient &= right_layer[:, 1:].logical_not() & existence[:, 1:] # rhs2l
    swapping_spot = agree_orient & joint_layer.logical_not()
    physical_joint = agree_orient & joint_layer
    bool_pads = torch.zeros(batch_size, 1, dtype = physical_joint.dtype, device = physical_joint.device)
    rhs_exist = torch.cat([bool_pads, physical_joint], dim = 1)
    lhs_exist = rhs_exist.logical_not()
    lhs_exist &= existence
    rhs_exist &= existence
    lhs_seq_idx = torch.cumsum(lhs_exist, dim = 1)
    rhs_seq_idx = lhs_seq_idx.roll(1, 1) 
    lhs_seq_idx *= lhs_exist
    rhs_seq_idx *= rhs_exist
    max_len = lhs_seq_idx.max() + 1
    lhs_helper = lhs_seq_idx, max_len, True
    rhs_helper = rhs_seq_idx, max_len, True
    return lhs_helper, rhs_helper, physical_joint, swapping_spot, bool_pads

def condense_helper(existence_or_start,
                    as_existence      = False,
                    offset            = None,
                    get_rid_of_last_k = 0):
    seq_idx = torch.cumsum(existence_or_start, 1)

    if offset is not None:
        if isinstance(offset, int):
            seq_idx += offset
        else:
            seq_idx += offset[:, None]

    # 0 used as a dump
    if get_rid_of_last_k:
        max_idx = seq_idx[:, -1:] - get_rid_of_last_k
        seq_idx[seq_idx > max_idx] = 0
        
    if as_existence:
        seq_idx *= existence_or_start
        
    max_len = seq_idx.max() + 1

    return seq_idx, max_len, as_existence

def condense_left(hidden, helper,
                  out_len    = None,
                  get_cumu   = False,
                  get_indice = False,
                  skip_dump0 = True):
    # from hidden redirected by seq_idx to zeros, throw away 1:
    hidden_shape = list(hidden.shape)
    seq_idx, max_len, as_existence = helper
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
        seq_idx = seq_idx.unsqueeze(dim = -1)
        cumula_shape = hidden_shape[:-1] + [1]
    else:
        cumula_shape = hidden_shape

    base = torch.zeros(*hidden_shape, device = hidden.device, dtype = hidden.dtype)
    if as_existence:
        base = base.scatter(1, seq_idx.expand_as(hidden), hidden)
    else:
        base = base.scatter_add(1, seq_idx.expand_as(hidden), hidden)

    # 0 used as a dump
    base = base[:, skip_dump0:] if truncate is None else base[:, skip_dump0:truncate]

    if get_cumu:
        cumu = torch.zeros(*cumula_shape, device = hidden.device, dtype = seq_idx.dtype)
        cumu.scatter_add_(1, seq_idx, torch.ones_like(seq_idx))
        cumu = cumu[:, skip_dump0:] if truncate is None else cumu[:, skip_dump0:truncate]
        if as_existence:
            cumu = cumu > 0

    num = get_indice + get_cumu
    if num == 0:
        return base
    if num == 1:
        return base, cumu if get_cumu else seq_idx
    return base, cumu, seq_idx

def release_left(base, seq_idx):
    base_shape = base.shape
    if len(base_shape) == 3:
        batch_size, seq_len, model_dim = base_shape
        left = torch.zeros(batch_size, 1, model_dim, dtype = base.dtype, device = base.device)
        seq_idx = seq_idx.expand(batch_size, -1, model_dim)
    else:
        batch_size, seq_len = base_shape
        left = torch.zeros(batch_size, 1, dtype = base.dtype, device = base.device)
    base = torch.cat([left, base], dim = 1)
    return base.gather(1, seq_idx) # all non-dump index > 0

def shuffle_some_from(new_size, base, existence):
    buffer = base.masked_select(existence).reshape(-1, base.shape[-1])
    buffer_size = buffer.shape[0]
    indices = torch.randint(high = buffer_size, size = (new_size,), device = base.device)
    some = torch.index_select(buffer, 0, indices)

    batch_ids = torch.where(existence)[0]
    batch_ids = torch.index_select(batch_ids, 0, indices)

    continuous = indices[1:] - indices[:-1] == 1
    continuous &= batch_ids[1:] == batch_ids[:-1]

    return some, continuous

def blocky_softmax(sections,
                   src_logits,
                   dst_logits = None,
                   final_src_logits = None,
                   pad_first = True,
                   map_pad_weights_to_zero = True,
                   use_ones_as_values = False):
    # 0 in sections indicate a dump
    # [1 2 2 3 0] -> [0 1 2 3]
    idx = sections.unsqueeze(dim = 2)
    expanded_idx = idx.expand_as(src_logits)
    bs, sl, eb = src_logits.shape
    domain_sl = sections.max() + 1
    if dst_logits is not None:
        padded = torch.zeros(bs, 1, eb, device = src_logits.device)
        padded = torch.cat([padded, dst_logits] if pad_first else [dst_logits, padded], dim = 1)
        padded = torch.gather(padded, 1, expanded_idx)
        src_logits = src_logits + padded # empty pad section will pollute src_logits
    exp_logits = src_logits.exp()
    zer_dst_logits = torch.zeros(bs, domain_sl, eb, device = src_logits.device)
    exp_dst_logits = zer_dst_logits.scatter_add(1, expanded_idx, exp_logits)
    exp_dst_logits = torch.gather(exp_dst_logits, 1, expanded_idx)
    if map_pad_weights_to_zero: # polluted src_logits yield zero weights
        if pad_first:
            existence = idx != 0
        else:
            existence = idx != sections.max(dim = 1).values[:, None, None]
        exp_logits = existence * exp_logits
    weights = exp_logits / exp_dst_logits
    if use_ones_as_values:
        from utils.shell_io import byte_style
        print(byte_style('[WARNING: use_ones_as_values only for debugging!!!]', '1'))
        print(byte_style('[+PROMPT: final_dst_logits should have either 0 or 1]', '3'))
        final_src_logits = torch.ones_like(weights)
    if final_src_logits is None:
        return weights
    final_dst_logits = zer_dst_logits.scatter_add(1, expanded_idx, weights * final_src_logits)
    final_dst_logits = final_dst_logits[:, 1:] if pad_first else final_dst_logits[:, :-1]
    return weights, final_dst_logits

def blocky_max(sections, values, has_batch_dim = True):
    m = sections.max() + 1
    if has_batch_dim:
        _, s = sections.shape
        s_range = torch.arange(s, device = sections.device)[None, :]
        m_range = torch.arange(m, device = sections.device)[None, :, None]
        mask = sections[:, None, :].repeat(1, m, 1) == m_range # [b, m, s]
        m_dim, s_dim = 1, 2
    else:
        s, = sections.shape
        s_range = torch.arange(s, device = sections.device)
        m_range = torch.arange(m, device = sections.device)[:, None]
        mask = sections[None, :].repeat(m, 1) == m_range
        m_dim, s_dim = 0, 1
    any_mask = mask.any(s_dim, keepdim = True)
    m_values = mask * values.unsqueeze(m_dim) # [b, m, s] * [b, 1, s]
    # _, indices = m_values.max(s_dim, keepdim = True)
    # print(indices)
    # print(m_values.argmax(s_dim, keepdim = True))# may have more than one...
    mask = s_range == m_values.argmax(s_dim, keepdim = True)
    mask &= any_mask
    return mask.any(m_dim)

def birnn_fwbw(embeddings, pad = None, existence = None):
    bs, sl, ed = embeddings.shape
    half_dim = ed >> 1
    if pad is None:
        pad = torch.ones(bs, 1, half_dim, dtype = embeddings.dtype, device = embeddings.device)
    else:
        pad = pad.repeat(bs, 1, 1) # not efficient as expand
    embeddings = embeddings.view(bs, sl, 2, half_dim)
    fw = embeddings[:, :, 0]
    bw = embeddings[:, :, 1]
    if existence is not None:
        within_seq = existence.unsqueeze(dim = 2)
        fw = torch.where(within_seq, fw, pad)
        bw = torch.where(within_seq, bw, pad)
    fw = torch.cat([pad, fw], dim = 1)
    bw = torch.cat([bw, pad], dim = 1)
    return fw, bw

def fencepost(fw, bw, splits):
    batch = torch.arange(fw.shape[0], device = fw.device)[:, None]
    fw = fw[batch, splits]
    fw_diff = fw[:, 1:] - fw[:, :-1]
    bw = bw[batch, splits]
    bw_diff = bw[:, :-1] - bw[:, 1:]
    return torch.cat([fw_diff, bw_diff], dim = 2)

def batch_insert(base, indice, subject = None):
    # base [b, s], obj [b, s-]
    batch_size, batch_len, emb_size = base.shape
    bs, extra_len = indice.shape
    assert batch_size == bs, 'batch_size should match'
    batch_range = torch.arange(batch_size, device = base.device)[:, None]
    extended_len = batch_len + extra_len

    inc_start = torch.zeros(batch_size, extended_len, dtype = indice.dtype, device = base.device)
    inc_start.scatter_add_(1, indice, torch.ones_like(indice)) # mark base (0) and subject (1) 224..
    # inc_start[batch_range, indice] += 1 does not work
    base_inc = inc_start.cumsum(dim = 1) # i-th insetion  0011222.. 
    base_inc = base_inc[:, :batch_len]
    base_inc += torch.arange(batch_len, device = base.device)[None] # base position 0134678..
    base_inc = base_inc.unsqueeze(dim = 2)

    new_base = torch.zeros(batch_size, extended_len, emb_size, dtype = base.dtype, device = base.device)
    new_base = new_base.scatter_add(1, base_inc.expand_as(base), base)
    if subject is None:
        return new_base
    
    subj_inc = torch.arange(extra_len, dtype = indice.dtype, device = base.device)[None]
    subj_inc = subj_inc + indice # 224 -> 236
    subj_inc.unsqueeze_(dim = 2)
    return new_base.scatter_add(1, subj_inc.expand_as(subject), subject)

def bool_start_end(continuous):
    batch_size, batch_len = continuous.shape
    seq_idx = torch.arange(batch_len, device = continuous.device)
    seq_idx = continuous * seq_idx
    max_idx = seq_idx.argmax(dim = 1)
    seq_idx[continuous.logical_not()] = batch_len
    min_idx = seq_idx.argmin(dim = 1)
    continuity = continuous.any(dim = 1)
    batch_dim = torch.arange(batch_size, device = continuous.device)
    return batch_dim[continuity], min_idx[continuity], max_idx[continuity]

def scatter_2d_add(src, dst, index):
    # flatten args to scatter_add
    sb, ss, se = src.shape
    db, ds, de = dst.shape
    src = src.reshape(sb * ss, se)
    frm = dst.reshape(db * ds, de)
    assert se == de
    flat = index[:, :, 0] * ds + index[:, :, 1]
    flat = flat.view(-1, 1).expand_as(src)
    return frm.scatter_add(0, flat, src).view(db, ds, de)