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
        if 'label' in field_v2is:
            field_v2is = field_v2is.copy()
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
            extra_text_helper = extra_text_helper(text, device)
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

    def at_idx(self, idx, factor, length):
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
        return field_columns


# class DynamicCrossDataset(LengthOrderedDataset):
#     def __init__(self,
#                  dir_join,
#                  trees,
#                  field_v2is,
#                  device,
#                  factors  = None,
#                  min_len  = 0,
#                  max_len  = None,
#                  extra_text_helper = None):