from data.backend import LengthOrderedDataset, np, torch, token_first
from utils.file_io import read_data
from time import time
from tqdm import tqdm
from data.delta import E_XDIM
from data.cross import unzip_xlogit, targets
from data.trapezoid import trapezoid_to_layers
from itertools import zip_longest

fields = 'token', 'tag'

class CrossDataset(LengthOrderedDataset):
    def __init__(self,
                 dir_join,
                 prefix,
                 field_v2is,
                 device,
                 factors  = None,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None,
                 train_indexing_cnn = False):
        columns = {}
        factored_indices = {}
        if 'label' in field_v2is:
            field_v2is = field_v2is.copy()
            if train_indexing_cnn:
                field_v2is.pop('label')
            field_v2is['xtype'] = (len(E_XDIM), int)

        for field, v2i in token_first(field_v2is):
            start = time()
            if field in fields:
                print(f'Load Dataset {prefix}.{field}', end = ' ', flush = True)
                if field == 'token':
                    column, lengths, text = read_data(dir_join(f'{prefix}.word'), v2i, True)
                else:
                    column = read_data(dir_join(f'{prefix}.{field}'), v2i, False)
                columns[field] = column
            else:
                for factor in factors:
                    if factor not in factored_indices:
                        factored_indices[factor] = read_data(dir_join(f'{prefix}.index.{factor}'), (2, int)) # 2-byte/512 > 500
                    with tqdm(total = len(lengths), desc = f'Load Dataset {prefix}.{field}.{factor}') as qbar:
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
                    
            print(f'in {time() - start:.2f}s')
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
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        cat_joint = 'target' in self.heads
        pad_len = 2 if cat_joint else 1 # for cnn_indexing

        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, np.int32)
                max_len = np.max(lengths) + pad_len # BOS and EOS
                offsets = (max_len - lengths) // 2
                field_columns['offset'] = offsets
                tensor = lengths
            elif field in fields: # token or tags
                tensor = np.zeros([batch_size, max_len], np.int32)
                for i, (values, offset, length) in enumerate(zip(column, offsets, lengths)):
                    end = offset + length
                    tensor[i, offset:end] = values
            elif field == 'seq_len':
                seq_len = zip(*zip_longest(*column, fillvalue = 0))
                seq_len = np.asarray(list(seq_len))
                segments = np.max(seq_len, axis = 0)
                field_columns['seq_len']  = seq_len
                field_columns['segments'] = segments
                continue
            else:
                dtype = np.int32 if field in ('label', 'target') else np.bool
                size_adjust = 0
                is_right_field = field == 'right'
                if field != 'joint':
                    sizes = seq_len
                    slens = segments + pad_len
                elif cat_joint:
                    sizes = seq_len[:, :-1] - 1
                    slens = segments + pad_len
                    size_adjust -= 1
                else:
                    sizes = seq_len[:, :-1] - 1
                    slens = segments[:-1] - 1 + pad_len
                tensor = np.zeros([batch_size, np.sum(slens) + size_adjust], dtype)
                for i, inst_size in enumerate(zip(column, sizes)):
                    l_start = 0
                    for (slen, layer, size) in zip(slens, *inst_size):
                        if size > 0:
                            start = l_start + pad_len
                            end   = start + size
                            tensor[i, start:end] = layer
                            if is_right_field:
                                tensor
                            l_start += slen

            field_columns[field] = tensor

        if cat_joint: # target -= 1 would make any difference
            slens = segments + pad_len
            tensor = np.zeros([batch_size, np.sum(slens)], np.bool)
            for i, layer_lens in enumerate(seq_len):
                l_start = 0
                for (slen, size) in zip(slens, layer_lens):
                    if size > 0:
                        start = l_start + pad_len
                        end   = start + size
                        tensor[i, start:end] = True
                        l_start += slen
            field_columns['existence'] = tensor

        for f, column in field_columns.items():
            if f in ('length', 'offset', 'target', 'segments', 'seq_len'):
                dtype = torch.int32
            elif f in ('token', 'tag', 'label'):
                dtype = torch.long
            else:
                dtype = torch.bool
            field_columns[f] = torch.as_tensor(column, dtype = dtype, device = self._device)
        return field_columns
