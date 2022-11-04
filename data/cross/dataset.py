import torch
from data.cross import Signal
from data.dataset import LengthOrderedDataset, np, read_signals
from data.dataset import pad_tag_like_nil, pad_tag_like_bos_eos
from data.dataset import pad_label_like_nil, pad_label_like_bos_eos
from data.dataset import erect_joint_less, fill_bool_tensor
from data.dataset import binary_signals, checkin_cache
from data.cross.multib import total_fence, continuous_fence
from data.continuous.binary import X_RGT
from utils.types import F_RANDOM
from itertools import zip_longest
from bidict import bidict

from collections import namedtuple
HybridM = namedtuple('HybridM', 'non_random, msub, cache')

def b_compare(head, data):
    return head[:3] == data[:3]

def m_compare(head, data):
    return head[:2] == data[:2]

class DiscontinuousDataset(LengthOrderedDataset):
    def __init__(self,
                 binary,
                 reader,
                 fileids,
                 from_fn,
                 v2is,
                 factor,
                 esub,
                 msub,
                 b_pad_shuffle_or_m_fence_intra_inter,
                 min_gap = 0,
                 min_len = 0,
                 max_len = None,
                 extra_text_helper = None,
                 self_check_i2vs = None):

        w2i, t2i, l2i = v2is
        (length, token, tag, signals,
         text) = read_signals(w2i, t2i, fileids, reader, Signal, from_fn, esub, False)
        if min_gap:
            for eid, signal in enumerate(signals):
                if signal.gap >= min_gap:
                    length[eid] = -1

        heads = 'tree', 'token'
        if factor is None:
            label = extra = signal_kwargs = None
        else:
            if binary:
                _heads = 'tag', 'label', 'xtype', 'joint', 'swap'
                factor, extra = self.reset_binary_factor(factor, esub, msub, initialize = len(length))
                paddings, ply_shuffle = b_pad_shuffle_or_m_fence_intra_inter
                ply_shuffle_option = 2 if ply_shuffle else 0
                signal_kwargs = dict(l2i = l2i, ply_shuffle_option = ply_shuffle_option, ply_shuffle_offset = 1)
                b_pad_shuffle_or_m_fence_intra_inter = paddings, ply_shuffle
            else:
                _heads = 'tag', 'label', 'space', 'disco', 'space_'
                factor, extra = self.reset_multib_factor(factor, esub, msub, initialize = len(length))
                signal_kwargs = dict(l2i = l2i, append_space = 1)
            heads = heads + _heads
            label = 'label'

        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, w2i)

        super().__init__(heads, label, length, factor, min_len, max_len, extra_text_helper)
        self._args = token, tag, signals, signal_kwargs, binary, b_pad_shuffle_or_m_fence_intra_inter, self_check_i2vs, extra

    def reset_multib_factor(self, factor, esub, msub, *height_intra_inter_rates, initialize = False):
        non_random = bidict({e:f for e, (f, p) in enumerate(factor.items()) if f != F_RANDOM and p > 0})
        if has_static_n := (msub in (0, 1) and sum(non_random)):
            hy_factor = {}
            for e, f in non_random.items():
                hy_factor[e] = factor[f] * (1 - esub)
                hy_factor[e + has_static_n] = factor[f] * esub
            if rand_p := factor.get(F_RANDOM):
                hy_factor[-1] = rand_p * (1 - esub)
                hy_factor[-2] = rand_p * esub
        else:
            esub_factor = {0: 1 - esub, 1: esub}

        if initialize:
            assert not height_intra_inter_rates
            if has_static_n:
                cache_fn = lambda n: tuple([None] * (has_static_n << 1) for _ in range(n))
            else: # pure random
                return esub_factor, msub
            return hy_factor, HybridM(non_random, msub, cache_fn(initialize))
        assert len(height_intra_inter_rates) == 3
        if not has_static_n:
            extra = msub
        elif isinstance((old := self._args[-1]), HybridM):
            if old.non_random == non_random:
                cache = old.cache
            else:
                cache = cache_fn(len(self._args[0]))
            extra = HybridM(non_random, msub, cache)
        self._args = self._args[:-3] + (((self._args[-3][0],) + height_intra_inter_rates), self._args[-2], extra)
        super()._reset_factors(hy_factor if has_static_n else esub_factor)

    def at_idx(self, idx, factor, helper_outputs):
        token, tag, signals, signal_kwargs, binary, _, _, extra = self._args
        signal = signals[idx]
        sample = [signal.tree, token[idx]]
        if extra is not None:
            sample.append(tag[idx])
            if binary:
                sample.extend(binary_signals(factor, idx, extra, lambda frac, esub, msub = 0: signal.binary(frac, esub, msub, **signal_kwargs), b_compare))
            elif isinstance(extra, HybridM):
                breakpoint()
                if factor < 0:
                    sample.extend(signal.multib(F_RANDOM, factor == -2, extra.msub, **signal_kwargs))
                else:
                    cache = extra.cache[idx]
                    if cache[factor] is None:
                        n = len(extra.non_random)
                        if esub := (idx >= n):
                            f = extra.non_random[idx - n]
                        else:
                            f = extra.non_random[idx]
                        checkin_cache(cache, factor, signal.multib(f, esub, extra.msub, **signal_kwargs), m_compare)
                    sample.extend(cache[factor])
            else:
                sample.extend(signal.multib(F_RANDOM, factor, extra, **signal_kwargs))
        return tuple(sample)

    def _collate_fn(self, batch, length, segment):
        _, _, _, _, binary, b_pad_shuffle_or_m_fence_intra_inter, self_check_i2vs, extra = self._args
        indice_args = dict(device = self.device, dtype = torch.long)
        field_columns = dict(length = length)
        max_token_len = length.max()

        if binary:
            paddings, ply_shuffle = b_pad_shuffle_or_m_fence_intra_inter
        else:
            paddings = None

        if paddings:
            max_token_len += 2 # BOS and EOS
            offset = (max_token_len - length) // 2
            field_columns['offset'] = torch.as_tensor(offset, **indice_args)
            field_columns['token'] = torch.as_tensor(pad_tag_like_bos_eos(batch.token, max_token_len, offset, *paddings['token']), **indice_args)
        else:
            field_columns['token'] = torch.as_tensor(pad_tag_like_nil(batch.token, max_token_len, int(binary)), **indice_args) # binary need 1 for engineering

        if extra is None:
            field_columns['tree'] = batch.tree
        else:
            max_tag_len = length.max()
            batch_size  = len(length)
            field_columns['segment'] = segment
            field_columns['batch_segment'] = batch_segment = segment.max(0)
            bool_args = dict(dtype = torch.bool, device = self.device)

            if paddings:
                field_columns['tag'] = torch.as_tensor(pad_tag_like_bos_eos(batch.tag, max_tag_len, offset, *paddings['tag']), **indice_args)
            else:
                field_columns['tag'] = torch.as_tensor(pad_tag_like_nil(batch.tag, max_tag_len, int(binary)), **indice_args)

            if paddings:
                field_columns['label'] = torch.as_tensor(pad_label_like_bos_eos(batch.label, batch_segment, offset, *paddings['label']), **indice_args)
            else:
                field_columns['label'] = torch.as_tensor(pad_label_like_nil(batch.label, batch_segment, int(binary)), **indice_args)

            if binary:
                if paddings:
                    field_columns['xtype'] = torch.as_tensor(pad_label_like_bos_eos(batch.xtype, batch_segment, offset, X_RGT, 0, X_RGT), dtype = torch.uint8, device = self.device)
                    breakpoint()
                else:
                    field_columns['xtype'] = torch.as_tensor(pad_label_like_nil(batch.xtype, batch_segment, 1), dtype = torch.uint8, device = self.device)
                    field_columns['joint'] = sig = torch.zeros(batch_size, batch_segment.sum(), **bool_args)
                    fill_bool_tensor(erect_joint_less(batch.joint, batch_segment, 0), sig, True, indice_args)
                    if ply_shuffle:
                        layers_of_swappers = []
                        for layer_len in batch_segment + 1:
                            layer = np.tile(np.arange(layer_len), batch_size)
                            layer.shape = (batch_size, layer_len)
                            layers_of_swappers.append(layer)
                        for bid, swap_layers in enumerate(batch.swap):
                            for swap_layer, layer_tensor in zip(swap_layers, layers_of_swappers):
                                for group_idx in swap_layer:
                                    layer_tensor[bid, group_idx] = np.random.permutation(group_idx)
                        field_columns['ply_shuffle'] = torch.as_tensor(np.concatenate(layers_of_swappers, axis = 1), **indice_args)

                if self_check_i2vs:
                    pass
            else:
                continuous_fence_only, inter_2d, intra_rate, inter_rate = b_pad_shuffle_or_m_fence_intra_inter
                field_columns['disco_2d_intra_rate'] = intra_rate
                field_columns['disco_2d_inter_rate'] = inter_rate
                assert not paddings, 'DM does not cover paddings.'
                field_columns['space'] = torch.as_tensor(pad_label_like_nil(batch.space_, batch_segment), **indice_args)

                # 1d: space as labels; disco as labels; [b, s+]; fence [b, s-]
                split_segment = []
                con_split_column = []
                dis_layer_column = []
                # 2d: [b, s++] & [b, 2ds--]
                shape = []
                components = []
                if inter_2d:
                    condensed_cnt = [0] * batch_size
                    condensed_max_layer_size = []
                    condense_layer = {}
                    condense_exclude = {bid: None for bid, b in enumerate(batch.disco) if sum(bool(l) for l in b) == 1}
                    condense_last_disco = {}
                    condense_kinship = {}
                for src_lid, (l_space, l_disco) in enumerate(zip(zip_longest(*batch.space, fillvalue = []), zip_longest(*batch.disco, fillvalue = {}))): # all layer slices [(), ] [(), ]
                    batch_layer_disco = [] # same dim with space
                    batch_layer_split = [] # splitting points for continuous constituents
                    max_split_len = max_comp_len = max_comp_size = 0
                    for space_layer, disco_set in zip(l_space, l_disco): # every layer for a parse
                        split_count, split_layer = continuous_fence(space_layer, disco_set) if continuous_fence_only else total_fence(space_layer)
                        batch_layer_split.append(split_layer)
                        if split_count > max_split_len:
                            max_split_len = split_count
                    comp_batch = []
                    l_condnse_layer = {}
                    for src_bid, disco_set in enumerate(l_disco):
                        disco_children = []
                        if disco_set:
                            num_comp_size = len(disco_set)
                            if num_comp_size > max_comp_size:
                                max_comp_size = num_comp_size
                            for ds in disco_set.values():
                                disco_children += ds
                            num_comp_len = len(disco_children)
                            if num_comp_len > max_comp_len:
                                max_comp_len = num_comp_len
                            comp_batch.append((disco_set, disco_children))

                            if inter_2d:
                                if src_bid in condense_exclude:
                                    assert condense_exclude[src_bid] is None
                                    condense_exclude[src_bid] = src_lid
                                else:
                                    dst_lid = condensed_cnt[src_bid]
                                    condensed_cnt[src_bid] += 1
                                    l_condnse_layer[src_bid] = dst_lid, num_comp_len
                                    if dst_lid < len(condensed_max_layer_size):
                                        cmls = condensed_max_layer_size[dst_lid]
                                        if num_comp_len > cmls:
                                            condensed_max_layer_size[dst_lid] = num_comp_len
                                    else:
                                        condensed_max_layer_size.append(num_comp_len)
                                    if src_bid in condense_last_disco:
                                        for oid, last_src_lid in enumerate(reversed(condense_last_disco[src_bid])):
                                            if src_bid not in condense_kinship:
                                                kinship = []
                                                condense_kinship[src_bid] = {oid: kinship}
                                            elif oid not in condense_kinship[src_bid]:
                                                condense_kinship[src_bid][oid] = kinship = []
                                            else:
                                                kinship = condense_kinship[src_bid][oid]
                                            kinship.append(disco_inter_gen(batch.disco[src_bid][last_src_lid],
                                                                           batch.space[src_bid][last_src_lid:src_lid],
                                                                           disco_set,
                                                                           src_lid - last_src_lid > inter_2d))
                                if src_bid in condense_last_disco:
                                    condense_last_disco[src_bid].append(src_lid)
                                else:
                                    condense_last_disco[src_bid] = [src_lid]
                        batch_layer_disco.append(disco_children)
                    if l_condnse_layer:
                        condense_layer[src_lid] = l_condnse_layer
                    split_segment.append(max_split_len + 1)
                    con_split_column.append(batch_layer_split)
                    dis_layer_column.append(batch_layer_disco)

                    comp_layer = []
                    for disco_set, disco_children in comp_batch:
                        disco_children.sort()
                        disco_children = {y:x for x,y in enumerate(disco_children)}
                        comp_layer.append([[disco_children[d] for d in ds] for ds in disco_set.values()])
                    components.append(comp_layer)
                    shape.append((len(comp_batch), max_comp_size, max_comp_len))

                field_columns['split_segment'] = split_segment
                field_columns['dis_disco'] = dis_disco = torch.zeros(batch_size, batch_segment.sum(), **bool_args)
                field_columns['con_split'] = con_split = torch.zeros(batch_size, sum(split_segment), **bool_args)

                fill_bool_tensor(fill_bool_layers(con_split_column, split_segment, True), con_split, True, indice_args)
                fill_bool_tensor(fill_bool_layers(dis_layer_column, batch_segment),       dis_disco, True, indice_args)
                if any(components):
                    if inter_2d and any(condensed_cnt):
                        field_columns['inter_disco'] = InterLayerDisco(condensed_cnt, condensed_max_layer_size, condense_layer, condense_exclude, condense_kinship)
                    start = 0
                    # dis_slice_shape = []
                    comp = np.zeros(sum(b*l*l for b, _, l in shape), dtype = np.bool)
                    for (bz, cz, cl), comps in zip(shape, components):
                        if bz:
                            end = start + bz * cl * cl
                            cp = comp[start:end].reshape(bz, cl, cl)
                            for bid, bpz in enumerate(comps):
                                for cpz in bpz:
                                    cps = cp[bid, cpz]
                                    cps[:, cpz] = True
                                    cp[bid, cpz] = cps
                            start = end
                    field_columns['dis_component'] = torch.tensor(comp, device = self.device)
        return field_columns


def component_segment(shape):
    shape = np.array([[0, 0, 0]] + shape)
    return np.cumsum(np.prod(shape, 1))

class InterLayerDisco:
    lhs_dim = rhs_dim = None
    tensor_args = {}

    def __init__(self, condensed_cnt, condensed_max_layer_size, condense_layer, condense_exclude, condense_kinship):
        bid_s2d = {}
        for dst_bid, (src_bid, cnt) in enumerate(sorted(enumerate(condensed_cnt), key = lambda x: -x[1])):
            if cnt == 0: break
            bid_s2d[src_bid] = dst_bid
        b_dim = len(bid_s2d)
        s_dim = sum(condensed_max_layer_size)
        dst_volumn = [[0, {}] for _ in condensed_max_layer_size] # batch vs. layer
        for layer in condense_layer.values():
            for src_bid, (dst_lid, dst_len) in layer.items():
                dst_volumn[dst_lid][0] += 1
                dst_volumn[dst_lid][1][bid_s2d[src_bid]] = dst_len
        for eid, (n, d) in enumerate(dst_volumn):
            dst_volumn[eid] = n, tuple(d[k] for k in sorted(d))
        layer_exclude = {}
        for src_bid, lid in condense_exclude.items():
            if lid in layer_exclude:
                layer_exclude[lid].add(src_bid)
            else:
                layer_exclude[lid] = {src_bid}
        ordered_kinship = {}
        for src_bid, o_layers in condense_kinship.items():
            for oid, layers in o_layers.items():
                if oid in ordered_kinship:
                    kinship = ordered_kinship[oid]
                else:
                    kinship = ordered_kinship[oid] = []
                for dst_lid, sm_gen in enumerate(layers):
                    if dst_lid < len(kinship):
                        layer = kinship[dst_lid]
                    else:
                        layer = ([], [], [])
                        kinship.append(layer)
                    bl, sl, ml = layer
                    for si, mi in sm_gen:
                        bl.append(bid_s2d[src_bid])
                        sl.append(si)
                        ml.append(mi)

        self._args = b_dim, s_dim, bid_s2d, condensed_max_layer_size, condense_layer, layer_exclude, dst_volumn, ordered_kinship
        if self.lhs_dim and self.rhs_dim:
            self.create_base(self.lhs_dim, self.rhs_dim, **self.tensor_args)

    def __str__(self):
        b_dim, s_dim, _, condensed_max_layer_size, condense_layer, _, _, _ = self._args
        s = f'D.samples: {b_dim}, Max.comp: {s_dim}\n  L.sizes: '
        return s + f'{condensed_max_layer_size}\n  Op. {condense_layer}'

    def create_base(self, lhs_dim, rhs_dim, **tensor_args):
        batch_dim, seq_dim = self._args[:2]
        batch_seq_dim = batch_dim * seq_dim + 1 #0 as a dump
        self._lhs = torch.zeros(batch_seq_dim, lhs_dim, **tensor_args)
        self._rhs = torch.zeros(batch_seq_dim, rhs_dim, **tensor_args)

    def store(self, src_lid, lhs, rhs):
        _, seq_dim, bid_s2d, condensed_max_layer_size, condense_layer, layer_exclude, _, _ = self._args
        if src_lid not in condense_layer:
            assert src_lid in layer_exclude
            return
        lb, ls, le = lhs.shape
        rb, rs, re = rhs.shape
        assert lb == rb and ls == rs
        bs = lb * ls
        lhs = lhs.reshape(bs, le)
        rhs = rhs.reshape(bs, re)
        index = np.zeros(bs, dtype = np.long)
        layer = condense_layer[src_lid]
        bid_a2r = {a:r for r,a in enumerate(sorted(layer.keys()|layer_exclude.get(src_lid, set())))}
        for src_bid, (dst_lid, n_disco) in layer.items():
            rel_bid = bid_a2r[src_bid] * ls
            dst_bid = bid_s2d[src_bid] * seq_dim + sum(condensed_max_layer_size[:dst_lid]) + 1
            for i in range(n_disco):
                index[rel_bid + i] = dst_bid + i
        index = torch.tensor(index, device = lhs.device)
        index.unsqueeze_(-1)
        self._lhs.scatter_add_(0, index.expand(bs, le), lhs)
        self._rhs.scatter_add_(0, index.expand(bs, re), rhs)

    def get(self, max_order = 0):
        device = self.tensor_args.get('device')
        batch_dim, seq_dim, _, cmls, _, _, dstv, ok = self._args
        lhs = self._lhs[1:].reshape(batch_dim, seq_dim, -1)
        rhs = self._rhs[1:].reshape(batch_dim, seq_dim, -1)
        for o in range(min(max_order, max(ok.keys())) + 1):
            ds = 0
            for sl, ml, (bw, sv), (bv, mv), kl in zip(cmls, cmls[o+1:], dstv, dstv[o+1:], ok[o]):
                assert bw >= bv, 'batch volumn must decrease'
                dm = ds + sl
                de = dm + ml
                sl = torch.arange(sl, device = device)
                ml = torch.arange(ml, device = device)
                sv = torch.tensor(sv[:bv], device = device)
                mv = torch.tensor(mv[:bv], device = device)
                sl.unsqueeze_(0); sv.unsqueeze_(1)
                ml.unsqueeze_(0); mv.unsqueeze_(1)
                mt = (sl < sv).unsqueeze(2) & (ml < mv).unsqueeze(1) # [bs, sl, ml]
                mt[kl] = False
                sm = lhs[:bv, ds:dm], rhs[:bv, dm:de], mt
                ms = lhs[:bv, dm:de], rhs[:bv, ds:dm], mt.transpose(1, 2)
                # print(cmls, dstv)
                # print(ds, dm, de, o)
                # print([x.shape for x in sm])
                # print([x.shape for x in ms])
                # breakpoint()
                yield sm, ms
                ds = dm

def enumerate_values(disco_set):
    map = []
    for bvs in disco_set.values():
        map.extend(bvs)
    map.sort()
    return {v:k for k, v in enumerate(map)}

def disco_inter_gen(bottom_disco, layers_of_space, top_disco, get_all):
    top_map = enumerate_values(top_disco)
    bottom_map = enumerate_values(bottom_disco)
    if get_all:
        for tv in top_map.values():
            for bv in bottom_map.values():
                yield (bv, tv)
    else:
        for bk, bvs in bottom_disco.items():
            for space in layers_of_space[1:]:
                bk = space[bk]
            for tvs in top_disco.values():
                if get_all or bk in tvs:
                    tk = top_map[bk]
                    for bv in bvs:
                        yield (bottom_map[bv], tk)

def fill_bool_layers(sample_layers, tensor_seg, remant = False):
    start = 0
    positive = []
    for seg_len, layer in zip(tensor_seg, sample_layers):
        end = start + seg_len
        for bid, seq in enumerate(layer):
            if seq:
                positive += ((bid, start + x) for x in seq)
            elif remant:
                positive += [(bid, start), (bid, start + 1)]
        start = end
    return positive