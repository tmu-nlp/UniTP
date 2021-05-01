from data.backend import LengthOrderedDataset, np, torch, token_first
from utils.file_io import read_data
from utils.shell_io import byte_style
from tqdm import tqdm
from data.delta import E_XDIM
from data.cross.binary import unzip_xlogit, targets, unzip_swaps
from data.trapezoid import trapezoid_to_layers
from itertools import zip_longest
from utils.types import O_HEAD, S_EXH

fields = 'token', 'tag'

class StaticCrossDataset(LengthOrderedDataset):
    def __init__(self,
                 dir_join,
                 prefix,
                 field_v2is,
                 device,
                 factors  = None,
                 min_len  = 0,
                 max_len  = None,
                 swapper  = False,
                 min_gap  = 0,
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

        field_v2is = token_first(field_v2is)
        num_fields_left = -1
        field_names = []
        for f, _ in field_v2is:
            if f in fields:
                field_names.append(f)
                num_fields_left += 1
            else:
                num_factors = len(factors)
                field_names.append(f + f'({num_factors})')
                num_fields_left += num_factors

        tqdm_prefix = f"Load {prefix.title().ljust(5, '-')}Set: "
        with tqdm(desc = tqdm_prefix + ', '.join(field_names)) as qbar:
            for field, v2i in field_v2is:
                qbar.desc = tqdm_prefix + f'\033[32m' + ', '.join((f + '\033[m') if f.startswith(field) else f for f in field_names)
                if field in fields: # token /tag / ftag / finc
                    if field == 'token':
                        column, lengths, text = read_data(dir_join(f'{prefix}.word'), v2i, True)
                        qbar.total = num_fields_left * len(lengths)
                    else:
                        column = read_data(dir_join(f'{prefix}.{field}'), v2i, qbar = qbar)
                    columns[field] = column
                else:
                    for factor in factors:
                        if factor not in factored_indices:
                            factored_indices[factor] = read_data(dir_join(f'{prefix}.index.{factor}'), (2, int)) # 2-byte/512 > 500
                        
                        flatten = read_data(dir_join(f'{prefix}.{field}.{factor}'), v2i)
                        if field == 'label':
                            column = []
                            for layer, sizes in zip(flatten, factored_indices[factor]):
                                column.append(trapezoid_to_layers(layer, sizes, sizes, big_endian = False))
                                qbar.update(1)
                            columns[(field, factor)] = column
                        elif field == 'xtype':
                            c_joint, c_right, c_direc = [], [], []
                            if train_indexing_cnn:
                                c_target = []
                            for layer, sizes in zip(flatten, factored_indices[factor]):
                                lr, lj, ld = unzip_xlogit(sizes, layer)
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
            assert len(lengths) == lid + 1
        assert all(len(lengths) == len(col) for col in columns.values())

        for factor, column in factored_indices.items():
            columns[('seq_len', factor)] = column

        if train_indexing_cnn:
            heads = 'token', 'seq_len', 'right', 'joint', 'target'
        else:
            heads = 'token', 'tag', 'seq_len', 'label', 'right', 'direc', 'joint'
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, device, c2i)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)

        self._swap = {}
        if swapper:
            for factor in factors:
                swap = []
                with open(dir_join(f'{prefix}.swap.{factor}')) as fr:
                    for line in fr:
                        swap.append(line)
                self._swap[factor] = swap
                assert len(lengths) == len(swap)
        self._swap_cache = None
        self._except_head = swapper == S_EXH

        self._columns = columns
        self._device = device

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = {}
        for field, column in self._columns.items():
            if isinstance(field, tuple) and field[1] == factor:
                sample[field[0]] = column[idx]
            else:
                sample[field]    = column[idx]
        sample['length'] = length

        if self._swap:
            swap = self._swap[factor][idx]
            if isinstance(swap, str):
                swap = unzip_swaps(swap, 1)
                self._swap[factor][idx] = swap # update cache
            if self._swap_cache is None:
                self._swap_cache = [(factor, swap)]
            else:
                self._swap_cache.append((factor, swap))
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
            elif field in fields: # token or tags
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
            field_columns[f] = torch.as_tensor(column, dtype = dtype, device = self._device)

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
            field_columns['swap'] = [torch.as_tensor(x, dtype = torch.long, device = self._device) for x in swappers]
        field_columns['offset'] = pad_len
        return field_columns


# from data.io import distribute_jobs
from data.multib.dataset import fill_layers
from data.cross.multib import F_RANDOM
class DynamicCrossDataset(LengthOrderedDataset):
    def __init__(self,
                 tree_keepers,
                 device,
                 factors = None,
                 min_len = 0,
                 max_len = None,
                 min_gap  = 0,
                 extra_text_helper = None,
                 c2i = None,
                 num_threads = 0):

        text = []
        lengths = []
        static_signals = []
        for tk in tree_keepers:
            assert tk.has_signals
            wd = tk.word
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
                lengths.append(len(wd))
            if factors is None:
                static_signals.append((wi, ti) + tk.stratify(F_RANDOM))
            if extra_text_helper is not None:
                text.append(wd)

        heads = 'token', 'tag', 'label', 'space', 'disco'
        if factors is None:
            heads = heads[:-1]
            tree_keepers = static_signals
            lines = ['Load ' + byte_style('static D.M. treebank', '3')]
        else:
            balanced_prob = factors['balanced']
            original_prob = 1 - balanced_prob
            train_factors = {}
            lines = ' F\Balanced'
            if balanced_prob:
                lines += '          Yes'
            if original_prob:
                lines += '      No (Origin without _SUB)'
            lines = ['Load ' + byte_style('dynamic D.M. treebank', '7'), byte_style(lines, '2')]
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
            factors = train_factors
        print('\n'.join(lines))
            
        self._keepers_heads = tree_keepers, heads
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, device, c2i)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)
        self._device = device

    def at_idx(self, idx, factor, length, helper_outputs):
        tree_keepers, heads = self._keepers_heads
        tk = tree_keepers[idx]
        if factor is None:
            signals = tk
        else:
            signals = tk.word_tag + tk.stratify(factor[1:], factor[0] == '+')
        sample = {h:s for h, s  in zip(heads, signals)}
        sample['length'] = length
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
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
                field_columns['segment'] = torch.tensor(segment, device = self._device)
                field_columns['seg_length'] = torch.tensor(seg_len, device = self._device).transpose(0, 1)
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
                for l_space, l_disco in zip(zip_longest(*space_column, fillvalue = []), zip_longest(*column, fillvalue = {})): # all layer slices [(), ] [(), ]
                    batch_layer_disco = [] # same dim with space
                    batch_layer_split = [] # splitting points for continuous constituents
                    max_space_len = 0
                    for space_layer, disco_set in zip(l_space, l_disco): # every layer for a parse
                        count = 0
                        split_layer = []
                        for lhs, rhs in zip(space_layer, space_layer[1:] + [-1]):
                            if lhs in disco_set:
                                continue
                            else:
                                count += 1
                            if lhs != rhs:
                                split_layer.append(count)
                        if split_layer:
                            split_layer.insert(0, 0)
                        batch_layer_split.append(split_layer)
                        if count > max_space_len:
                            max_space_len = count
                    comp_batch = []
                    max_comp_len = 0
                    max_comp_size = 0
                    for disco_set in l_disco:
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
                        batch_layer_disco.append(disco_children)
                    split_segment.append(max_space_len + 1)
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
                field_columns['dis_disco'] = torch.tensor(fill_bool_layers(batch_size, dis_layer_column, segment), device = self._device)
                field_columns['con_split'] = torch.tensor(fill_bool_layers(batch_size, con_split_column, split_segment, True), device = self._device)
                if any(components):
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
                    field_columns['dis_component'] = torch.tensor(comp, device = self._device)

            field_columns[field] = torch.as_tensor(tensor, dtype = torch.long, device = self._device)
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