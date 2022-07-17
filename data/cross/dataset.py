from data.backend import LengthOrderedDataset, np, torch, token_first
from utils.file_io import read_data
from utils.shell_io import byte_style
from tqdm import tqdm
from data.delta import E_XDIM
from data.cross.binary import unzip_xlogit, targets, disco_tree, TreeKeeper
from data.cross.binary import unzip_and_double_swaps_p1, double_swaps_p1
from data.trapezoid import trapezoid_to_layers
from itertools import zip_longest
from utils.types import O_HEAD, S_EXH, F_RAND_CON, F_RAND_CON_SUB, F_RAND_CON_MSB, device

simple_fields = 'token', 'tag'

beta_string = lambda beta: f'Beta({", ".join(f"{x:.2e}" for x in beta)})'
subs_string = lambda x, y: f'sub({x * 100:.0f}%)  msb({y * 100:.0f}%)'

class BinaryDataset(LengthOrderedDataset):
    def __init__(self,
                 dir_join,
                 prefix,
                 field_v2is,
                 factors  = None,
                 min_len  = 0,
                 max_len  = None,
                 min_gap  = 0,
                 ply_shuffle = None,
                 extra_text_helper = None,
                 train_indexing_cnn = False):

        columns = {}
        factored_indices = {}
        has_char = 'char' in field_v2is
        has_label = 'label' in field_v2is
        if has_char or has_label:
            field_v2is = field_v2is.copy()
            c2i = field_v2is.pop('char')[1] if has_char else None
            if has_label:
                if train_indexing_cnn:
                    field_v2is.pop('tag')
                    field_v2is.pop('label')
                field_v2is['xtype'] = (len(E_XDIM), int)
        sub_ratio = factors.pop(F_RAND_CON_SUB, 0)
        self._more_sub = factors.pop(F_RAND_CON_MSB, 0)
        if fully_randomize := factors.pop(F_RAND_CON, False):
            factors = {O_HEAD: 1}
            _, w2i = field_v2is['token']
            _, t2i = field_v2is['tag']
            _, l2i = field_v2is['label']
            v2is = w2i, t2i, l2i
        field_v2is = token_first(field_v2is)

        num_fields_left = -1
        field_names = []
        for f, _ in field_v2is:
            if f in simple_fields:
                field_names.append(f)
                num_fields_left += 1
            else:
                num_factors = len(factors)
                field_names.append(f + f'({num_factors})')
                num_fields_left += num_factors

        from utils.str_ops import StringProgressBar
        sbar = StringProgressBar(field_names, ', ')
        tqdm_prefix = f"Load {prefix.title().ljust(5, '-')}Set: "
        with tqdm(desc = tqdm_prefix + str(sbar)) as qbar:
            for field, v2i in field_v2is:
                if fully_randomize and field != 'xtype':
                    v2i = None
                if field in simple_fields: # token /tag / ftag / finc
                    if field == 'token':
                        column, lengths, text = read_data(dir_join(f'{prefix}.word'), v2i, True)
                        num_samples = len(lengths)
                        sbar.update(2, total = num_factors * num_samples)
                        sbar.update(3, total = num_factors * num_samples)
                        qbar.total = num_fields_left * num_samples
                    else:
                        column = read_data(dir_join(f'{prefix}.{field}'), v2i, qbar = qbar)
                    columns[field] = column
                    qbar.desc = tqdm_prefix + str(sbar.update(field, 1.0))
                else:
                    for factor in factors:
                        if factor not in factored_indices:
                            factored_indices[factor] = read_data(dir_join(f'{prefix}.index.{factor}'), (2, int)) # 2-byte/512 > 500
                        
                        flatten = read_data(dir_join(f'{prefix}.{field}.{factor}'), v2i)
                        if field == 'label':
                            column = []
                            for line, sizes in zip(flatten, factored_indices[factor]):
                                column.append(trapezoid_to_layers(line, sizes, sizes, big_endian = False))
                                qbar.update(1)
                                qbar.desc = tqdm_prefix + str(sbar.update(2))
                            columns[(field, factor)] = column
                        elif field == 'xtype':
                            c_joint, c_right, c_direc = [], [], []
                            if train_indexing_cnn:
                                c_target = []
                            for line, sizes in zip(flatten, factored_indices[factor]):
                                lr, lj, ld = unzip_xlogit(sizes, line)
                                c_right.append(lr)
                                c_joint.append(lj)
                                if train_indexing_cnn:
                                    target = []
                                    for lri, lji in zip(lr, lj + [[]]):
                                        target.append(targets(lri, lji))
                                    c_target.append(target)
                                else:
                                    c_direc.append(ld)
                                qbar.update(1)
                                qbar.desc = tqdm_prefix + str(sbar.update(3))
                            columns[('right', factor)] = c_right
                            columns[('joint', factor)] = c_joint
                            if train_indexing_cnn:
                                columns[('target', factor)] = c_target
                            else:
                                columns[('direc', factor)] = c_direc
                        else:
                            raise ValueError('Unknown field: ' + field)
        
        if min_gap:
            with open(dir_join(f'{prefix}.gap')) as fr:
                for lid, gap in enumerate(fr):
                    if int(gap) < min_gap:
                        lengths[lid] = 0
            assert num_samples == lid + 1
        assert all(num_samples == len(col) for col in columns.values())

        for factor, column in factored_indices.items():
            columns[('seq_len', factor)] = column

        if train_indexing_cnn:
            heads = 'token', 'seq_len', 'right', 'joint', 'target'
        else:
            heads = 'token', 'tag', 'seq_len', 'label', 'right', 'direc', 'joint'
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, c2i)
        self._swap_cache = None
        self._except_head = ply_shuffle == S_EXH
        if fully_randomize:
            fields = simple_fields + tuple((x, O_HEAD) for x in ('label', 'right', 'joint', 'direc'))
            keepers = []
            if isinstance(fully_randomize, tuple):
                func = beta_string(fully_randomize)
            else:
                func = 'Uniform'
            desc = f'Convert {prefix.title()}Set to {num_samples} dynamic TreeKeepers w/'
            desc += ' ' if ply_shuffle else 'o '
            desc += 'ply shuffle.'
            print(byte_style(func, 6), '|', byte_style(subs_string(sub_ratio, self._more_sub), 3))
            with StringProgressBar(desc).update(total = num_samples) as sbar:
                for i in range(num_samples):
                    btm, tpd, err = disco_tree(*(columns[x][i] for x in fields))
                    keepers.append(TreeKeeper(btm, tpd, v2is))
                    sbar.update()
                    assert err is None
            columns = keepers
            if 0 < sub_ratio < 1:
                factors = {False: 1 - sub_ratio, True: sub_ratio}
            else:
                factors = sub_ratio == 1
            self._swap = ply_shuffle
            self._beta = fully_randomize
        else:
            self._swap = {}
            if ply_shuffle:
                for factor in factors:
                    swap = []
                    with open(dir_join(f'{prefix}.swap.{factor}')) as fr:
                        for line in fr:
                            swap.append(line)
                    self._swap[factor] = swap
                    assert num_samples == len(swap)
            
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)
        self._columns = columns

    def reset_factors(self, factors):
        if factors.get(F_RAND_CON):
            self._more_sub = msb = factors[F_RAND_CON_MSB]
            self._beta = beta = factors[F_RAND_CON]
            sub_ratio = factors[F_RAND_CON_SUB]
            print(byte_style(beta_string(beta), 6), '|', byte_style(subs_string(sub_ratio, msb), 3))
            if 0 < sub_ratio < 1:
                factors = {False: 1 - sub_ratio, True: sub_ratio}
            else:
                factors = sub_ratio == 1
        self._reset_factors(factors)

    def __cache_swap(self, *swap):
        if self._swap_cache is None:
            self._swap_cache = [swap]
        else:
            self._swap_cache.append(swap)

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = dict(length = length)
        if not isinstance(factor, str):
            tk = self._columns[idx]
            if isinstance(self._beta, tuple):
                alpha, beta = self._beta
                rho = np.random.beta(alpha, beta)
            else:
                rho = F_RANDOM
            sample['token'], sample['tag'] = tk.word_tag
            (label, sample['right'], sample['joint'], sample['direc'],
             swap) = tk.stratify(rho, factor, self._more_sub)
            sample['label'] = label
            sample['seq_len'] = [len(x) for x in label]
            if self._swap:
                self.__cache_swap(factor, double_swaps_p1(swap))
            return sample
        for field, column in self._columns.items():
            if isinstance(field, tuple) and field[1] == factor:
                sample[field[0]] = column[idx]
            else:
                sample[field]    = column[idx]

        if self._swap:
            swap = self._swap[factor][idx]
            if isinstance(swap, str):
                swap = unzip_and_double_swaps_p1(swap)
                self._swap[factor][idx] = swap # update cache
            self.__cache_swap(factor, swap)
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        cat_joint = 'target' in self.heads
        pad_len = 2 if cat_joint else 1 # for cnn_indexing

        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, np.int32)
                max_len = np.max(lengths) + pad_len # <nil>s as BOS
                tensor = lengths
            elif field in simple_fields: # token or tags
                tensor = np.zeros([batch_size, max_len], np.int32)
                for i, (values, length) in enumerate(zip(column, lengths)):
                    tensor[i, pad_len: pad_len + length] = values
            elif field == 'seq_len':
                seq_len = zip(*zip_longest(*column, fillvalue = 0))
                seq_len = np.asarray(list(seq_len))
                segments = np.max(seq_len, axis = 0)
                segments += pad_len
                field_columns['seq_len']  = seq_len
                field_columns['segments'] = segments
                continue
            else:
                dtype = np.int32 if field in ('label', 'target') else np.bool
                size_adjust = 0
                if field != 'joint':
                    sizes = seq_len
                    slens = segments
                elif cat_joint:
                    sizes = seq_len[:, :-1] - 1
                    slens = segments
                    size_adjust -= 1
                else:
                    sizes = seq_len[:, :-1] - 1
                    slens = segments[:-1] - 1
                tensor = np.zeros([batch_size, np.sum(slens) + size_adjust], dtype)
                for i, inst_size in enumerate(zip(column, sizes)):
                    l_start = 0
                    for (slen, layer, size) in zip(slens, *inst_size):
                        if size > 0:
                            start = l_start + pad_len
                            end   = start + size
                            tensor[i, start:end] = layer
                            l_start += slen

            field_columns[field] = tensor

        tensor = np.zeros([batch_size, np.sum(segments)], np.bool)
        for i, layer_lens in enumerate(seq_len):
            l_start = 0
            for (slen, size) in zip(segments, layer_lens):
                if size > 0:
                    start = l_start + pad_len
                    end   = start + size
                    tensor[i, start:end] = True
                    l_start += slen
        field_columns['existence'] = tensor

        for f, column in field_columns.items():
            if f in ('length', 'target', 'segments', 'seq_len'):
                dtype = torch.int32
            elif f in ('token', 'tag', 'label'):
                dtype = torch.long
            else:
                dtype = torch.bool
            field_columns[f] = torch.as_tensor(column, dtype = dtype, device = device)

        if self._swap_cache: # TODO swap for emb, label
            swappers = []
            for size in segments:
                layer = np.arange(size)
                layer = np.tile(layer, batch_size)
                layer.shape = (batch_size, size)
                swappers.append(layer)
            for sid, (factor, swap) in enumerate(self._swap_cache):
                if self._except_head and factor == O_HEAD:
                    continue
                for layer, layer_tensor in zip(swap, swappers):
                    for (group_idx, group_ctn) in layer:
                        np.random.shuffle(group_ctn)
                        layer_tensor[sid, group_idx] = group_ctn
            self._swap_cache = None # clear batch
            field_columns['swap'] = [torch.as_tensor(x, dtype = torch.long, device = device) for x in swappers]
        field_columns['offset'] = pad_len
        return field_columns

from data.io import sorting_order, sort_by_order
from data.multib.dataset import fill_layers
from data.cross.multib import total_fence, continuous_fence, F_RANDOM
class MultibDataset(LengthOrderedDataset):
    def __init__(self,
                 tree_keepers,
                 factors = None,
                 extra_text_helper = None,
                 c2i = None,
                 continuous_fence_only = True,
                 min_len = 0,
                 max_len = None,
                 min_gap  = 0,
                 inter_2d = 0):

        text = []
        lengths = []
        static_signals = []
        for tk in tree_keepers:
            assert tk.has_signals
            tt = tk.text
            wi, ti = tk.word_tag
            if None in wi:
                print('Wrong vocab?')
                lengths.append(-1)
            elif None in ti:
                print('Wrong vocab?')
                lengths.append(-1)
            elif min_gap and tk.gaps < min_gap:
                lengths.append(-1)
            else:
                lengths.append(len(tt))
            if factors is None:
                static_signals.append((wi, ti) + tk.stratify(F_RANDOM))
            text.append(tt)

        heads = 'token', 'tag', 'label', 'space', 'disco'
        if factors is None:
            heads = heads[:-1]
            tree_keepers = static_signals
            lines = ['Load ' + byte_style('static D.M. treebank', '3')]
            rate = 0
        else:
            factors, lines, rate = self.__reset_and_show_factors(factors, 'Load ')
        print('\n'.join(lines))

        order = sorting_order(text)
        lengths, tree_keepers = (sort_by_order(order, x) for x in (lengths, tree_keepers))
            
        self._keepers_heads = tree_keepers, heads, rate
        if extra_text_helper:
            text = sort_by_order(order, text)
            extra_text_helper = extra_text_helper(text, c2i)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)
        self._fence_2d = continuous_fence_only, inter_2d
        InterLayerDisco.tensor_args.update(device = device)

    def __reset_and_show_factors(self, factors, prefix):
        balanced_prob = factors['balanced']
        more_sub_prob = factors['more_sub']
        original_prob = 1 - balanced_prob
        train_factors = {}
        lines = ' F\Balanced'
        if balanced_prob:
            lines += '          Yes'
        if original_prob:
            lines += '      No (Origin without _SUB)'
        if more_sub_prob: lines += f' [+{100 * more_sub_prob:.0f}% random _SUB]'
        lines = [prefix + byte_style('dynamic D.M. treebank', '7'), byte_style(lines, '2')]
        for factor, o_prob in factors['others'].items():
            line = ''
            prob = balanced_prob * o_prob
            if prob:
                train_factors['+'+factor] = prob
                line += f'{prob * 100:.0f}%'.rjust(9)
            prob = original_prob * o_prob
            if prob:
                train_factors['-'+factor] = prob
                line += f'{prob * 100:.0f}%'.rjust(18)
            if line:
                lines.append(f'  ::{ factor}::'.ljust(15) + line)
        return train_factors, lines, more_sub_prob

    def reset_factors(self, factors, inter_2d = 0):
        factors, lines, rate = self.__reset_and_show_factors(factors, 'Reset ')
        self._keepers_heads = self._keepers_heads[:2] + (rate,)
        self._fence_2d = self._fence_2d[0], inter_2d
        if inter_2d: lines.append(f'  Max. inter-height: {inter_2d}')
        print('\n'.join(lines))
        self._reset_factors(factors)

    def at_idx(self, idx, factor, length, helper_outputs):
        tree_keepers, heads, rate = self._keepers_heads
        tk = tree_keepers[idx]
        if factor is None:
            signals = tk
        else:
            signals = tk.word_tag + tk.stratify(factor[1:], factor[0] == '+', rate)
        sample = {h:s for h, s  in zip(heads, signals)}
        sample['length'] = length
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        continuous_fence_only, inter_2d = self._fence_2d
        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                tensor = lengths = np.asarray(column, np.int32)
                max_len = np.max(lengths)
            elif field in ('token', 'tag'):
                tensor = np.zeros([batch_size, max_len], np.int32)
                for i, (values, length) in enumerate(zip(column, lengths)):
                    tensor[i, :length] = values
            elif field == 'label':
                segment = []
                seg_len = []
                for layer in zip_longest(*column, fillvalue = []):
                    sl = [len(seq) for seq in layer]
                    segment.append(max(sl))
                    seg_len.append(sl)
                field_columns['segment'] = torch.tensor(segment, device = device)
                field_columns['seg_length'] = torch.tensor(seg_len, device = device).transpose(0, 1)
                tensor = fill_layers(column, segment, np.int32)
            elif field == 'space':
                tensor = fill_space_layers(batch_size, column, segment[:-1])
                space_column = column
            else:
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
                    condense_exclude = {bid: None for bid, b in enumerate(column) if sum(bool(l) for l in b) == 1}
                    condense_last_disco = {}
                    condense_kinship = {}
                for src_lid, (l_space, l_disco) in enumerate(zip(zip_longest(*space_column, fillvalue = []), zip_longest(*column, fillvalue = {}))): # all layer slices [(), ] [(), ]
                    batch_layer_disco = [] # same dim with space
                    batch_layer_split = [] # splitting points for continuous constituents
                    max_split_len = 0
                    for space_layer, disco_set in zip(l_space, l_disco): # every layer for a parse
                        split_count, split_layer = continuous_fence(space_layer, disco_set) if continuous_fence_only else total_fence(space_layer)
                        batch_layer_split.append(split_layer)
                        if split_count > max_split_len:
                            max_split_len = split_count
                    comp_batch = []
                    max_comp_len = 0
                    max_comp_size = 0
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
                                            kinship.append(disco_inter_gen(column[src_bid][last_src_lid],
                                                                           space_column[src_bid][last_src_lid:src_lid],
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
                field_columns['dis_disco'] = torch.tensor(fill_bool_layers(batch_size, dis_layer_column, segment), device = device)
                field_columns['con_split'] = torch.tensor(fill_bool_layers(batch_size, con_split_column, split_segment, True), device = device)
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
                            # dis_slice_shape.append((start, end, bz, cl))
                            start = end
                        # else:
                        #     dis_slice_shape.append(None)
                    # size = np.array(shape).prod(1)
                    # comp = np.zeros(size.sum(), dtype = np.bool)
                    # start = 0
                    # for sz, sp, comps in zip(size, shape, components): # layer
                    #     if sz > 0:
                    #         end = start + sz # layer -> [b-, c+, s-]
                    #         cp = comp[start:end].reshape(sp)
                    #         for bid, bpz in enumerate(comps):
                    #             for cid, cpz in enumerate(bpz):
                    #                 cp[bid, cid, cpz] = True
                    #         start = end
                    # field_columns['dis_slice'] = component_segment(shape) 
                    # field_columns['dis_shape'] = shape # [(layer, comp_x, comp_y)] [b, 3]
                    # field_columns['dis_slice_shape'] = dis_slice_shape
                    field_columns['dis_component'] = torch.tensor(comp, device = device)

            field_columns[field] = torch.as_tensor(tensor, dtype = torch.long, device = device)
        return field_columns

def fill_space_layers(batch_size, space_layers, tensor_seg):
    tensor = np.zeros([batch_size, sum(tensor_seg)], dtype = np.int32)
    start = 0
    for seg_len, layer in zip(tensor_seg, zip_longest(*space_layers, fillvalue = [])):
        end = start + seg_len
        for bid, seq in enumerate(layer):
            if seq:
                seq_len = len(seq)
                seq_end = start + seq_len
                tensor[bid, start:seq_end] = seq
                tensor[bid, start:seq_end] += 1
            else:
                tensor[bid, start] = 1
        start = end
    return tensor

def fill_bool_layers(batch_size, sample_layers, tensor_seg, remant = False):
    tensor = np.zeros([batch_size, sum(tensor_seg)], dtype = np.bool)
    start = 0
    for seg_len, layer in zip(tensor_seg, sample_layers):
        end = start + seg_len
        for bid, seq in enumerate(layer):
            if seq:
                tensor[bid, [start + x for x in seq]] = True
            elif remant:
                tensor[bid, [start, start + 1]] = True
        start = end
    return tensor

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