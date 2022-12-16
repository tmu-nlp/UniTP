from models.accp import torch, MultiStem
from models.utils import condense_helper, condense_left, blocky_softmax, blocky_max, bool_start_end
from models.utils import BiaffineAttention, LinearBinary
from models.types import fmax
from utils.shell_io import byte_style
from utils import do_nothing
from sys import stderr

try:
    import xccp_decode
    import numpy as np
    print(byte_style('With our cuda::xccp_decode✨', '2'))
    def predict_disco_sections_(disco_2d, layer_chunk, dis_batch_idx, dis_length, dis_indice, head_score, get_logits):
        (sections, comp_id, component, comp_check, final_thresholds,
         trials) = xccp_decode.predict_disco_sections(disco_2d, dis_length, dis_batch_idx, dis_indice.contiguous(), layer_chunk.contiguous(), head_score, 0.5, True, -1)
        layer_2d_logits = {}
        if get_logits:
            disco_2d = (x.cpu().numpy() for x in (dis_batch_idx, dis_length, disco_2d.type(torch.float16), comp_id, component, comp_check, final_thresholds, trials))
            for bid, dis_len, clear_2d, cid, comps, check, thresh, trial in zip(*disco_2d):
                cid = cid[:dis_len]
                n_comps = np.unique(cid).shape[0]
                comps = comps[:n_comps, :dis_len]
                clear_2d = clear_2d[:dis_len, :dis_len]
                fail = n_comps == dis_len and bool(thresh < 0)
                layer_2d_logits[int(bid)] = clear_2d, cid, comps, n_comps, fail, thresh, trial
        return sections, layer_2d_logits
except:
    print(byte_style('[HINT] CUDA module xccp_decode', '3'), 'is not compiled for full discontinuity matrix parallelism.', file = stderr)
    print('    -> try', byte_style('\'cd cuda; python setup.py install\''), file = stderr)
    xccp_decode = None

def continuous_chunk(bbt_zeros, continuous, chunk_logits):
    batch_dim, con_min, con_max = bool_start_end(continuous)
    con_left  = torch.cat([bbt_zeros, continuous], dim = 1)
    con_right = torch.cat([continuous, bbt_zeros], dim = 1)
    con_left[batch_dim, con_min] = True
    con_right[batch_dim, con_max + 1] = True
    fl_left  = condense_left(chunk_logits, condense_helper(con_left,  as_existence = True))
    fl_right = condense_left(chunk_logits, condense_helper(con_right, as_existence = True))
    return torch.where(fl_left * fl_right > 0, # same sign
                       torch.where(fl_left > 0,
                                   torch.where(fl_left > fl_right, fl_left, fl_right),
                                   torch.where(fl_left < fl_right, fl_left, fl_right)),
                       fl_left + fl_right) # oppo sign

def get_head_score(medoid, disco_2d, dis_length, get_dis_emb, head_logit):
    dis_batch_size, dis_max_children, _ = disco_2d.shape
    seq = torch.arange(dis_max_children, device = disco_2d.device)
    disco_2d[:, seq, seq] = 1
    if (dis_length < dis_max_children).any():
        in_seq = seq[None] < dis_length[:, None]
        in_seq = in_seq.unsqueeze(1) & in_seq.unsqueeze(2)
        disco_2d *= in_seq
    if medoid == 'unsupervised_head':
        head_score = get_dis_emb(head_logit).mean(dim = -1)
        if head_score.shape != (dis_batch_size, dis_max_children):
            breakpoint()
    elif medoid.startswith('max_'): # max_lhs & max_rhs
        head_score = disco_2d.sum(dim = 1 if medoid.endswith('lhs') else 2)
    elif medoid.endswith('most'): # leftmost & rightmost
        head_score = seq.flip(0) if medoid.startswith('left') else seq
        head_score += 1 # +1 for blocky_max
        head_score = head_score[None].repeat(dis_batch_size, 1)
    elif medoid == 'random': # random
        head_score = torch.rand(dis_batch_size, dis_max_children, device = disco_2d.device)
    else:
        raise ValueError('Unknown medoid: ' + medoid)
    return head_score

def predict_disco_sections(disco_2d, layer_chunk, dis_batch_idx, dis_length, dis_indice, head_score, get_logits, debug = False):
    if disco_2d.is_cuda and xccp_decode is not None:
        cuda_sections, cuda_layer_2d_logits = predict_disco_sections_(disco_2d, layer_chunk, dis_batch_idx, dis_length, dis_indice, head_score, get_logits)
        if debug:
            assert get_logits
            _, dis_max_children, _ = disco_2d.shape
            fb_comps = torch.ones(1, dis_max_children, dtype = torch.bool, device = dis_batch_idx.device)
            fb_comp_idx = torch.zeros(dis_max_children, dtype = dis_batch_idx.dtype, device = dis_batch_idx.device)

            dis_comps = []
            layer_2d_logits = {}
            for clear_2d, dis_len, bid, indice, score in zip(disco_2d, dis_length, dis_batch_idx, dis_indice, head_score):
                threshold = 0.5
                clear_2d = clear_2d[:dis_len, :dis_len] # no way: need var sizes
                b_clear_2d = clear_2d > threshold # clear_2d.mean() # mean ? softer? bad for ones; CPU or GPU?  inclusion matrix for comp
                in_deg  = b_clear_2d.sum(dim = 0)
                out_deg = b_clear_2d.sum(dim = 1)
                # import pdb; pdb.set_trace()
                fallback = False
                n_trails = 1
                if (in_deg != out_deg).any():
                    comps = fb_comps[:, :dis_len]
                    comp_idx = fb_comp_idx[:dis_len]
                    fallback = True
                else:
                    comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True) # no way: map_fn
                    trials = list(clear_2d.storage())
                    trials.sort(key = lambda x: abs(x - threshold), reverse = True)
                    while not (comps.sum(dim = 0) == 1).all():
                        n_trails += 1
                        if trials:
                            threshold = trials.pop()
                            b_clear_2d = clear_2d > threshold
                            comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True)
                        else:
                            threshold = -1.0
                            break
                    if threshold < 0:
                        # import pdb; pdb.set_trace()
                        comps = fb_comps[:, :dis_len]
                        comp_idx = fb_comp_idx[:dis_len]
                        fallback = True
                        n_trails += 1
                temp = tuple(x.cpu().numpy() for x in (clear_2d, comp_idx, comps))
                n_comps = np.unique(temp[1]).shape[0]
                temp = temp + (n_comps, fallback, threshold, n_trails)
                layer_2d_logits[int(bid)] = temp

                dis_b_max = blocky_max(comp_idx, score[:dis_len], False) # 0110 (for 57)
                comp_max_idx = comp_idx[dis_b_max] # 0101 -> 01 or 10
                indice = indice[:dis_len]
                dis_max_idx = indice[dis_b_max] # 4578 -> 57
                layer_chunk[bid, dis_max_idx] = True # 00101_11_1 -> 
                dis_comps.append((bid, dis_max_idx, comps, comp_max_idx, indice))
            sections = layer_chunk.cumsum(dim = 1) # 0011123444
            for bid, dis_max_idx, comps, comp_max_idx, indice in dis_comps:
                order = sections[bid, dis_max_idx][comp_max_idx]
                sections[bid, indice] = (order[:, None] * comps).sum(dim = 0)
            if (sections != cuda_sections).any():
                diff = (sections != cuda_sections).any(1)
                print(f'{diff.sum():d}')
                print(sections[diff])
                print('Cuda:')
                print(cuda_sections[diff])
                breakpoint()
        else:
            sections, layer_2d_logits = cuda_sections, cuda_layer_2d_logits
        
    return sections, layer_2d_logits

def append_negative(layers_of_disco_2d_positive,
                    layers_of_disco_2d_negative,
                    logits, con_space):
    existence = con_space > 0
    if con_space.shape[1] > 1:
        existence = existence.unsqueeze(dim = 1) & existence.unsqueeze(dim = 2)
        eq_space = con_space.unsqueeze(dim = 1) == con_space.unsqueeze(dim = 2)
        layers_of_disco_2d_positive.append(logits[eq_space & existence])
        if (ne_space := ~eq_space & existence).any():
            layers_of_disco_2d_negative.append(logits[ne_space])
    else:
        layers_of_disco_2d_positive.append(logits.squeeze(dim = 2)[existence])

def wrap_split(logits, exsitence, pad = fmax):
    b = logits.shape[0]
    batch_dim = torch.arange(b, device = logits.device)
    l = torch.ones (b, 1, dtype = logits.dtype, device = logits.device) * pad
    r = torch.zeros(b, 1, dtype = logits.dtype, device = logits.device)
    logits = torch.cat([l, logits, r], dim = 1)
    logits[batch_dim, exsitence.sum(dim = 1)] = pad
    return -logits # because of chunk == 1 - affinity


from models.types import rnn_module_type, discontinuous_attention_hint, activation_type, logit_type, fmin, fmax
from utils.types import orient_dim, hidden_dim, frac_2, frac_4, num_ori_layer, true_type, xccp_chunk_type, BaseType
E_1D_IN = ('unit', 'state', 'diff')
def is_valid_2d_form(x):
    x = x.split('.')
    if len(x) != 3:
        return False
    lhs, op, rhs = x
    if lhs not in E_1D_IN or rhs not in E_1D_IN:
        return False
    if op in ('biaff', 'biaff+b') or op.isdecimal() and int(op) > 0:
        return True
    return False
disco_2d_form = BaseType('diff.biaff.diff', is_valid_2d_form)
disco_1d_form = BaseType(1, as_index = True, default_set = E_1D_IN)
medoid = BaseType(0, as_index = True, default_set = ('unsupervised_head', 'max_lhs', 'max_rhs', 'leftmost', 'rightmost', 'random'))
stem_config = dict(space_dim           = orient_dim,
                   disco_1d_hidden_dim = orient_dim,
                   disco_1d_form       = disco_1d_form,
                   disco_1d_activation = activation_type,
                   disco_2d_activation = activation_type,
                   disco_2d_form       = disco_2d_form,
                   disco_2d_medoid     = medoid,
                   chunk_linear_dim    = xccp_chunk_type,
                   space_module        = rnn_module_type,
                   chunk_activation    = activation_type,
                   attention_hint      = discontinuous_attention_hint,
                   num_layers          = num_ori_layer,
                   drop_out            = frac_4,
                   rnn_drop_out        = frac_2,
                   trainable_initials  = true_type)

def _get_arg0(x):
    return x[0]

def _get_arg1(x):
    return x[1]

def _get_arg_(x):
    return MultiStem.diff_emb(x[2], x[3])

def _select(key, model_dim, state_dim):
    if key == 'unit':
        return model_dim, _get_arg0
    elif key == 'state':
        return state_dim, _get_arg1
    elif key == 'diff':
        return state_dim, _get_arg_

class SelectCache:
    def __init__(self, *x):
        self._diff = None
        self._x = x

    @property
    def disco_1d(self):
        x = SelectCache.fn(self._x)
        if SelectCache.fn is _get_arg_:
            self._diff = x
        return x

    def disco_2d(self, gn):
        lhs_fn, rhs_fn = fns = SelectCache.fns
        if _get_arg_ in fns:
            if self._diff is None:
                x = self._diff = _get_arg_(self._x)
            else:
                x = self._diff # fn applied in 1d
            if lhs_fn is rhs_fn:
                diff = gn(x)
                return (diff, diff)
            return (gn(x if fn is _get_arg_ else fn(self._x)) for fn in fns)
        if lhs_fn is rhs_fn:
            return (gn(lhs_fn(self._x)),) * 2
        return (gn(fn(self._x)) for fn in fns)

from models import StemOutput
class DiscoMultiStem(MultiStem):
    def __init__(self,
                 model_dim,
                 space_dim,
                 continuous_chunk_only,
                 disco_1d_form,
                 disco_1d_activation,
                 disco_2d_activation,
                 disco_2d_form,
                 disco_2d_medoid,
                 space_module,
                 disco_1d_hidden_dim,
                 chunk_linear_dim,
                 chunk_activation,
                 attention_hint,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 trainable_initials):
        super().__init__(model_dim,
                         space_dim,
                         chunk_linear_dim,
                         space_module,
                         None, False,
                         chunk_activation,
                         attention_hint,
                         num_layers,
                         drop_out,
                         rnn_drop_out,
                         trainable_initials)
        self._continuous_chunk_only = continuous_chunk_only
        dis_dim, SelectCache.fn = _select(disco_1d_form, model_dim, space_dim)
        self.predict_1d_disco = LinearBinary(dis_dim, disco_1d_hidden_dim, disco_1d_activation)

        self._medoid = disco_2d_medoid
        disco_2d_lhs, disco_2d_op, disco_2d_rhs = disco_2d_form.split('.')
        lhs_dim, lhs_select = _select(disco_2d_lhs, model_dim, space_dim)
        rhs_dim, rhs_select = _select(disco_2d_rhs, model_dim, space_dim)
        from data.cross.dataset import InterLayerDisco
        InterLayerDisco.lhs_dim = lhs_dim
        InterLayerDisco.rhs_dim = rhs_dim
        InterLayerDisco.tensor_args.update(dtype = self.predict_1d_disco.weight_in.weight.dtype)
        SelectCache.fns = lhs_select, rhs_select
        if disco_2d_op.startswith('biaff'):
            self.biaff_2d_disco = BiaffineAttention(lhs_dim, rhs_dim, disco_2d_op == 'biaff+b')
            def predict_2d_disco(lhs, rhs, existence = None):
                if is_2d := (existence is None):
                    rhs = rhs.transpose(1, 2)
                else:
                    lhs = lhs[:, :-1]
                    rhs = rhs[:, 1:]
                logits = self.biaff_2d_disco(self._stem_dp(lhs), self._stem_dp(rhs), not is_2d)
                return logits if is_2d else wrap_split(logits, existence)
        else:
            cat_dim = int(disco_2d_op)
            self.cat_2d_disco = LinearBinary(lhs_dim + rhs_dim, cat_dim, disco_2d_activation)
            def predict_2d_disco(lhs, rhs, existence = None):
                if is_2d := (existence is None):
                    batch_size, lhs_seq_dim, lhs_dim = lhs.shape
                    _,          rhs_seq_dim, rhs_dim = rhs.shape
                    lhs_hidden = lhs[:, :, None].expand(batch_size, lhs_seq_dim, rhs_seq_dim, lhs_dim)
                    rhs_hidden = rhs[:, None, :].expand(batch_size, lhs_seq_dim, rhs_seq_dim, rhs_dim)
                else:
                    lhs_hidden = lhs[:, :-1]
                    rhs_hidden = rhs[:, 1:]
                hidden = torch.cat([lhs_hidden, rhs_hidden], dim = -1)
                logits = self.cat_2d_disco(self._stem_dp(hidden), self._stem_dp)
                return logits if is_2d else wrap_split(logits, existence)
        if chunk_linear_dim == 0:
            self.predict_chunk = None
        self.predict_2d_disco = predict_2d_disco

    def forward(self,
                existence,
                unit_emb,
                supervision = None,
                disco_2d_intra_rate = 0,
                disco_2d_inter_rate = 0,
                get_disco_2d = False,
                **kw_args):
        batch_size, seg_len, model_dim = unit_emb.shape
        h0c0 = self.get_h0c0(batch_size)
        max_iter_n = seg_len + (seg_len >> 1) # 1.5 times
        teacher_forcing = supervision is not None
        batch_segment, segment = [], []
        batch_dim = torch.arange(batch_size, device = unit_emb.device)
        bbt_zeros = torch.zeros(batch_size, 1, dtype = torch.bool, device = existence.device)

        layers_of_u_emb = []
        layers_of_existence = []
        # training logits
        layers_of_chunk = [] # 101011
        layers_of_disco_1d = [] # 0000111100
        layers_of_disco_2d = [] # 1010x0101
        
        if teacher_forcing:
            space, dis_disco, inter_disco = supervision
            layers_of_disco_2d_positive = []
            layers_of_disco_2d_negative = []
            layers_of_inter_2d_negative = []
        else:
            layers_of_space = [] # 001132324
            layers_of_weight = []

        for l_cnt in range(max_iter_n):
            seq_len = existence.sum(dim = 1)
            layers_of_u_emb.append(unit_emb)
            layers_of_existence.append(existence)
            batch_segment.append(seg_len)
            segment.append(seq_len)

            disco_hidden, _ = self._chunk_emb(unit_emb, h0c0)
            fw, bw = self.pad_fwbw_hidden(disco_hidden, existence)
            disco_hidden = self._stem_dp(disco_hidden) # the order should be kept
            fw = self._stem_dp(fw)
            bw = self._stem_dp(bw)
            cache = SelectCache(unit_emb, disco_hidden, fw, bw)
            disco_1d_logits = self.predict_1d_disco(cache.disco_1d)
            layers_of_disco_1d.append(disco_1d_logits)
            if self.predict_chunk is None:
                chunk_logits = self.predict_2d_disco(*cache.disco_2d(do_nothing), existence)
            else:
                chunk_logits = self.predict_chunk(fw, bw) # local fw & bw for continuous
            layer_chunk_logits = chunk_logits # save for the next function

            longer_seq_idx = torch.arange(seg_len + 1, device = unit_emb.device)
            seq_idx = longer_seq_idx[None, :seg_len]
            if teacher_forcing:
                sections      = space    [l_cnt]
                discontinuous = dis_disco[l_cnt] # [b, s]
            else:
                discontinuous = disco_1d_logits > 0
                if discontinuous.shape != existence.shape:
                    print(f'WARNING: Invalid sections caused unmatched existence {l_cnt}', file = stderr, end = '')
                    break
                discontinuous &= existence
                discontinuous = torch.where(discontinuous.sum(dim = 1, keepdim = True) == 1,
                                            torch.zeros_like(discontinuous), discontinuous)
            
            if seg_len == 1:
                break # teacher forcing or a good model
            elif len(segment) > 1:
                prev, curr = segment[-2:]
                if (prev == curr).all():
                    break
                elif l_cnt == max_iter_n - 1:
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break
                
            dis_batch = discontinuous.any(dim = 1)
            if teacher_forcing and disco_2d_intra_rate:
                disco_1d_prob = self._sigmoid(disco_1d_logits) * disco_2d_intra_rate
                disco_1d_rand = torch.rand_like(disco_1d_prob, dtype = unit_emb.dtype) < disco_1d_prob
                disco_1d_rand &= existence
                if (con_con_rand_batch := dis_batch.logical_not() & disco_1d_rand.any(dim = 1)).any():
                    con_helper = condense_helper(disco_1d_rand[con_con_rand_batch], as_existence = True)
                    con_lhs, con_rhs = cache.disco_2d(lambda emb: condense_left(emb[con_con_rand_batch], con_helper))
                    con_space = condense_left(sections[con_con_rand_batch], con_helper)
                    append_negative(layers_of_disco_2d_positive,
                                    layers_of_disco_2d_negative,
                                    self.predict_2d_disco(con_lhs, con_rhs),
                                    con_space)

            if dis_batch.any():
                dis_batch_idx, = torch.where(dis_batch)
                dis_batch_size, = dis_batch_idx.shape
                continuous = existence & ~discontinuous
                if self._continuous_chunk_only:
                    chunk_logits = continuous_chunk(bbt_zeros, continuous, chunk_logits)

                if teacher_forcing and disco_2d_intra_rate:
                    con_rand = continuous & disco_1d_rand
                    dis_con_rand_batch = dis_batch & con_rand.any(dim = 1)
                    if dis_con_rand_batch.any():
                        dis_helper = condense_helper(discontinuous[dis_con_rand_batch], as_existence = True)
                        disco_lhs, disco_rhs = cache.disco_2d(lambda emb: condense_left(emb[dis_con_rand_batch], dis_helper))
                        
                        dis_con_rand = con_rand[dis_con_rand_batch]
                        con_helper = condense_helper(dis_con_rand, as_existence = True)
                        get_con_emb = lambda emb: condense_left(emb[dis_con_rand_batch], con_helper)
                        con_lhs, con_rhs = cache.disco_2d(get_con_emb)

                        dis_rand = condense_left(dis_con_rand, con_helper)
                        negative = self.predict_2d_disco(disco_lhs, con_rhs)
                        layers_of_disco_2d_negative.append(negative[dis_rand.unsqueeze(dim = 1).expand_as(negative)].reshape(-1))
                        negative = self.predict_2d_disco(con_lhs, disco_rhs)
                        layers_of_disco_2d_negative.append(negative[dis_rand.unsqueeze(dim = 2).expand_as(negative)].reshape(-1))
                        con_space = condense_left(sections[dis_con_rand_batch], con_helper)
                        append_negative(layers_of_disco_2d_positive,
                                        layers_of_disco_2d_negative,
                                        self.predict_2d_disco(con_lhs, con_rhs),
                                        con_space)
                    
                dis_exist = discontinuous[dis_batch_idx]
                dis_helper = condense_helper(dis_exist, as_existence = True)
                get_dis_emb = lambda emb: condense_left(emb[dis_batch_idx], dis_helper)
                card_lhs, card_rhs = cache.disco_2d(get_dis_emb)
                if disco_2d_inter_rate > 0 and hasattr(inter_disco, 'store'):
                    inter_disco.store(l_cnt, card_lhs, card_rhs)
                disco_2d_logits = self.predict_2d_disco(card_lhs, card_rhs)
            else:
                disco_2d_logits = None
            layers_of_chunk.append(chunk_logits)

            sub_emb = 0 # self._stem_dp(self._subject_bias())
            if self._subject_unit:  sub_emb = sub_emb + self._stem_dp(self._subject_unit(unit_emb))
            if self._subject_state: sub_emb = sub_emb + self._stem_dp(self._subject_state(disco_hidden))
            if self._subject_fw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_a(fw[:, 1:]))
            if self._subject_bw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_a(bw[:, :-1]))
            if self._subject_fw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_b(fw[:, :-1]))
            if self._subject_bw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_b(bw[:, 1:]))
            if self._subject_fw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_d(fw[:, 1:] - fw[:, :-1]))
            if self._subject_bw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_d(bw[:, :-1] - bw[:, 1:]))

            if teacher_forcing:
                if disco_2d_logits is not None:
                    layers_of_disco_2d.append(disco_2d_logits.reshape(-1))
            else:
                layer_len = seq_len[:, None]
                layer_chunk_logits[:, 0] = fmax
                layer_chunk_logits[batch_dim, seq_len] = fmax
                layer_chunk_logits[longer_seq_idx > layer_len] = fmin
                layer_chunk = (layer_chunk_logits > 0)[:, :-1]

                if disco_2d_logits is None:
                    sections = layer_chunk.cumsum(dim = 1)
                    # layers_of_slice.append((dis_slice_start, dis_slice_start))
                    # layers_of_shape.append(None)
                    layers_of_disco_2d.append({})
                else:
                    # layers_of_space.append(disco_2d.shape)
                    # dis_slice_end = dis_slice_start + disco_2d.shape.numel()
                    # layers_of_slice.append((dis_slice_start, dis_slice_end))
                    # 012345678
                    # 000011011 (disc.)
                    #_0101?11?1 v
                    #0010__1__1 v (result: layer_chunk)
                    #000011011_ ^
                    dis_length = dis_exist.sum(dim = 1)
                    dis_indice = condense_left(seq_idx.repeat(dis_batch_size, 1), dis_helper)
                    layer_chunk &= discontinuous.logical_not()
                    disco_2d_logits = self._sigmoid(disco_2d_logits) # need to be in a arange (not hard compress)
                    head_score = get_head_score(self._medoid, disco_2d_logits, dis_length, get_dis_emb, sub_emb)

                    sections, disco_2d = predict_disco_sections(disco_2d_logits, layer_chunk, dis_batch_idx, dis_length, dis_indice, head_score, get_disco_2d)
                    layers_of_disco_2d.append(disco_2d)
                sections = torch.where(seq_idx < layer_len, sections, torch.zeros_like(sections))
                layers_of_space.append(sections)
            # For the next layer,
            weights, unit_emb = blocky_softmax(sections, sub_emb, None, unit_emb)
            seg_len = unit_emb.shape[1]
            existence, _ = sections.max(dim = 1, keepdim = True)
            existence = seq_idx[:, :seg_len] < existence
            if not teacher_forcing:
                layers_of_weight.append(weights)

        segment = torch.stack(segment, dim = 1)
        embeddings = torch.cat(layers_of_u_emb,     dim = 1)
        existence  = torch.cat(layers_of_existence, dim = 1)
        chunk      = torch.cat(layers_of_chunk,     dim = 1) if layers_of_chunk else torch.zeros(batch_size, 0, dtype = unit_emb.dtype, device = unit_emb.device)
        disco_1d   = torch.cat(layers_of_disco_1d,  dim = 1)
        if teacher_forcing:
            if hasattr(inter_disco, 'inter'):
                for (ls, rm, mt), (lm, rs, tm) in inter_disco.get(): # mt &= backward error
                    negative = self.predict_2d_disco(ls, rm)
                    mt = mt & (torch.rand_like(negative) < (self._sigmoid(negative) * disco_2d_inter_rate))
                    layers_of_inter_2d_negative.append(negative[mt].reshape(-1))
                    negative = self.predict_2d_disco(lm, rs)
                    tm = tm & (torch.rand_like(negative) < (self._sigmoid(negative) * disco_2d_inter_rate))
                    layers_of_inter_2d_negative.append(negative[tm].reshape(-1))
            weight = space = None
            disco_2d = torch.cat(layers_of_disco_2d, dim = 0) if layers_of_disco_2d else None
            disco_2d_positive = torch.cat(layers_of_disco_2d_positive, dim = 0) if layers_of_disco_2d_positive else None
            disco_2d_negative = torch.cat(layers_of_disco_2d_negative, dim = 0) if layers_of_disco_2d_negative else None
            inter_2d_negative = torch.cat(layers_of_inter_2d_negative, dim = 0) if layers_of_inter_2d_negative else None
        else:
            disco_2d = layers_of_disco_2d
            disco_2d_positive = disco_2d_negative = inter_2d_negative = None
            if not layers_of_space:
                assert not layers_of_weight
                space  = torch.zeros(batch_size, 0, dtype = batch_dim.dtype, device = batch_dim.device)
                weight = torch.zeros(batch_size, 0, model_dim, dtype = unit_emb.dtype, device = unit_emb.device)
            else:
                space  = torch.cat(layers_of_space,  dim = 1)
                weight = torch.cat(layers_of_weight, dim = 1)

        return StemOutput(embeddings, existence, batch_segment, (weight, disco_1d, chunk, disco_2d, disco_2d_positive, disco_2d_negative, inter_2d_negative, space, segment))

    @property
    def message(self):
        messages = []
        if hasattr(self, 'biaff_2d_disco'):
            messages.append(f'Biaff.bias: {self.biaff_2d_disco.bias.bias.mean()}')
        if hasattr(super(), 'message') and (message := super().message):
            messages.append(message)
        if messages:
            return '\n'.join(messages)


# batch_insert(101011, 4444) -> 1010000011
#  001132324    001112334 [_-]=[01]
# 1010000011 -> _010_--_1:1 -> ****3232*
# or x inter
# batch_insert(_01011, 4444, 0110) -> 0010011011
# 001112334 001132324


# 000011110  4444
# 000011011  4455
# 4567 diff-> 111 -> 1000 -> 1111 -> 4444
# 4578 diff-> 121 -> 1010 -> 1122 -> 4455


multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(space_layer     = stem_config,
                  tag_label_layer = multi_class)

from models.loss import get_loss
from models.backend import ParsingOutputLayer
from utils.param_ops import change_key
class _DM(ParsingOutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'space_layer', 'stem_layer')
        super().__init__(DiscoMultiStem, *args, **kwargs)


# def discontinuous_hidden(dis_batch_idx, seq_idx, discontinuous):
    # 012345678
    # 001132432
    # 10101_11_1 (sup 1_1)
    # 1111001001 -> 101011
    # 1111100100 -> 101011
    # 101011 (min or avg)
    # 111100100
    # 1111100100
    #     ^cz
    # 1111001001
    # 001132324 (sup)
    # 012345678
    # 01238 4567 (con_indice & dis_indice)
    # 028   56 (blocky_max: boolean positions; lft|max) | 014   23 (cumsum & take out| concatenated & sorted)
    # 001112334 <- 112223445
    #     3232 (@4567)
    # 001132324 (sup)

    # dis_max_parents = dis_max_children >> 1
    # if dis_max_parents > 7:
    #     dis_max_parents = 7
    # dis_max_parents += 1 # make room for comparison [base, comp, comp, |]
    # # TODO - Q: How many components statistically are? Q: density of disco signals (layerwisely)
    # # CharLSTM; VisVote; map_ vmap?
    # start = time()
    # dis_u, dis_d, dis_v = torch.svd_lowrank(disco_2d, dis_max_parents) # [b, mp, mc], [b, mp]
    # dis_d_diff = dis_d[:, :dis_max_parents - 1] - dis_d[:, 1:dis_max_parents] # [b, mp]
    # dis_d_diff_argmax = dis_d_diff.argmax(dim = 1, keepdim = True)
    # seq_idx = torch.arange(dis_max_parents, device = dis_batch_idx.device)
    # dis_d[dis_d_diff_argmax < seq_idx[None]] = 0
    # clear_3d = torch.bmm(dis_d.unsqueeze(dim = 1) * dis_u, dis_v.transpose(1, 2))
    # svd_time = time() - start

        #                                       final      first     final      first
        # 4578
        # 1010 (7); 0101 (5) | *****1*1*
        # (32) -> 3030; 0202 -> 3232
        # 001112334 -> ****3232*
        # print(dis_max_idx)
    
    # dis_comps = [None] * dis_size
    # def disco_inject(dis_bid, con_bid, dis_len):
    #     clear_2d = clear_3d[dis_bid, :dis_len, :dis_len]
    #     b_clear_2d = clear_2d > 0.5
    #     in_deg  = b_clear_2d.sum(dim = 0)
    #     out_deg = b_clear_2d.sum(dim = 1)
    #     if (in_deg != out_deg).any() or (in_deg <= 1).any() or (out_deg <= 1).any():
    #         comps = fb_comps[:, :dis_len]
    #         comp_idx = fb_comp_idx[:dis_len]
    #     else:
    #         comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True) # no way: map_fn
    #         if not (comps.sum(dim = 0) == 1).all():
    #             comps = fb_comps[:, :dis_len]
    #             comp_idx = fb_comp_idx[:dis_len]

    #     dis_b_max = blocky_max(comp_idx, clear_2d.sum(dim = 0), False)
    #     comp_max_idx = comp_idx[dis_b_max] # 0101 -> 01 or 10
    #     dis_max_idx = dis_indice[dis_bid, :dis_len][dis_b_max]
    #     layer_chunk[con_bid, dis_max_idx] = True # 00101_11_1 -> 
    #     dis_comps[dis_bid] = dis_max_idx, comps, comp_max_idx
    #     return dis_bid

    # dis_idx = torch.arange(dis_size, device = dis_batch_idx.device)
    # dis_idx.map2_(dis_batch_idx, dis_length, disco_inject)

    # print(layer_chunk * 1)
        # num_comps, _ = comps.shape
        # comp_idx = (comp_dim[:num_comps, None] * comps).sum(dim = 0) # 0101 (or 1010)
    # min_sec, _ = sections.min(1)
    # if (min_sec > 1).any():
    #     print(torch.where(min_sec > 1))
    #     import pdb; pdb.set_trace()
    # print(sections)
    # print('sec')

    # def disco_assign(dis_bid, con_bid, dis_len):
    #     indice = dis_indice[dis_bid, :dis_len]
    #     dis_max_idx, comps, comp_max_idx = dis_comps[dis_bid]
    #     order = sections[con_bid, dis_max_idx][comp_max_idx]
    #     # print(con_bid, sections[con_bid, dis_max_idx])
    #     # print(comp_max_idx, sections[con_bid, dis_max_idx])
    #     # print((order[:, None] * comps).sum(dim = 0))
    #     sections[con_bid, indice] = (order[:, None] * comps).sum(dim = 0) # 3232 for 0101
    #     # print()
    #     return dis_bid
    # dis_idx.map2_(dis_batch_idx, dis_length, disco_assign)
 # 3232 for 0101
        # 0:0101  23 2
        # 1:1010  01 3 3232
        # 0:1010     3
        # 1:0101  10 2 3232
    # cate_time = time() - start - svd_time
    # with open('svd_time.csv', 'a+') as fw:
    #     fw.write(f'{dis_size},{dis_max_children},{svd_time},{cate_time}\n')

            # after sections
            # if (existence != ((unit_emb ** 2).sum(-1) > 0)).any():
            #     breakpoint()
                    # chunk_comp = ((seq_idx < layer_len) & (sections == 0)).any(dim = 1, keepdim = True)
                    # if chunk_comp.any():
                    #     print(torch.where(chunk_comp.squeeze(-1)))
                    #     print(torch.where(layer_chunk[:, 0].logical_not()))
                    #     breakpoint()
                    # sections += chunk_comp