from data.backend import LengthOrderedDataset
from utils.str_ops import write_ptr, delete_ptr, swap_ptr, is_numeric
import numpy as np
import torch

class CharDataset(LengthOrderedDataset):
    def __init__(self,
                 chars,
                 words,
                 field_v2is,
                 noise_specs,
                 factors, # origin replace_all insert [_1, _2] delete swap
                 device,
                 min_len  = 0,
                 max_len  = None):

        columns = {}
        for field, (_, v2i) in field_v2is.items():
            column  = []
            lengths = []
            is_digit = []
            for w in words:
                is_digit.append(is_numeric.fullmatch(w))
                lengths.append(len(w))
                column .append(tuple(v2i(c) for c in w))
            columns[field] = column
        assert all(len(lengths) == len(col) for col in columns.values())

        heads = 'token', 'first_validity', 'second_validity', 'noise_type'
        super().__init__(heads, lengths, factors, min_len, max_len, None)

        self._columns = columns
        self._itype_device = np.int32, device
        char_ids = tuple(v2i(c) for c in chars) # TODO is numeric bit
        char_gen = lambda: np.random.choice(char_ids)
        chars = lambda x: tuple(np.random.choice(char_ids) for _ in range(x)) # TODO make it numpy
        self._char_gens_specs = char_gen, chars, noise_specs, is_digit, words, v2i

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = {}
        char_gen, chars, noise_specs, is_digit, words, v2i = self._char_gens_specs
        for field, column in self._columns.items():
            char_ids        = column[idx]
            first_validity  = True
            second_validity = None
            if factor == 'replace_all' and length > 1:
                char_ids       = chars(length)
                first_validity = False
                factor = 'replace'
            elif factor == 'replace':
                replace = []
                for i_ in range(length):
                    if np.random.random() < noise_specs[factor]:
                        replace.append(i_)
                if replace:
                    char_ids       = list(char_ids)
                    first_validity = [True for _ in range(length)]
                    for i_, dst in enumerate(replace):
                        src = char_gen()
                        write_ptr(char_ids, i_, src, overwrite = True, as_unit = True)
                        first_validity[i_] = (src == dst)
                else:
                    factor = 'origin'
            elif factor == 'insert':
                insert = []
                for i_ in reversed(range(length + 1)): # +1 insert spot
                    if np.random.random() < noise_specs[factor]:
                        insert.append(i_)
                if insert:
                    char_ids       = list(char_ids)
                    first_validity = [True for _ in range(length)]
                    length += len(insert)
                    for i_ in insert:
                        write_ptr(char_ids,       i_, char_gen(), overwrite = False, as_unit = True)
                        write_ptr(first_validity, i_, False,      overwrite = False, as_unit = True)
                else:
                    factor = 'origin'
            elif factor == 'delete':
                delete = []
                for i_ in reversed(range(length)):
                    if np.random.random() < noise_specs[factor]:
                        delete.append(i_)
                if delete and len(delete) < length / 2:
                    char_ids        = list(char_ids)
                    second_validity = [True for _ in range(length)]
                    length -= len(delete)
                    for i_ in delete:
                        if i_ >= 1:
                            second_validity[i_-1] = False
                        if i_ + 1 < len(char_ids):
                            second_validity[i_+1] = False
                        delete_ptr(char_ids,        i_, 1)
                        delete_ptr(second_validity, i_, 1)
                    if is_digit[idx]:
                        second_validity = True
                    else:
                        second_validity = tuple(l | r for l, r in zip(second_validity, second_validity[1:]))
                else:
                    factor = 'origin'
            elif factor == 'swap': # swap_4.4 absolute short distance
                swap = []
                per, dis = noise_specs[factor]
                for i_ in range(length):
                    if np.random.random() < per:
                        distance = np.random.exponential(dis)
                        distance = int(distance + 1)
                        if distance < length:
                            swap.append(distance)
                if swap:
                    char_ids = list(char_ids)
                    for distance in swap:
                        start = np.random.randint(0, length - distance)
                        _end_ = start + distance
                        # if _end_ == length - 1:
                        #     import pdb; pdb.set_trace()
                        swap_ptr(char_ids, start, _end_, 1)
                    _validity = tuple(o == s for o,s in zip(column[idx], char_ids))
                    if is_digit[idx] or all(_validity):
                        first_validity = True
                        factor = 'origin'
                    else:
                        first_validity = _validity
                else:
                    factor = 'origin'

            if second_validity is None:
                if isinstance(first_validity, (list, tuple)):
                    second_validity = tuple(l & r for l, r in zip(first_validity, first_validity[1:]))
                else:
                    second_validity = first_validity

            sample[ field ]           = char_ids
            sample['first_validity']  = first_validity
            sample['second_validity'] = second_validity
            sample['noise_type']      = factor
            sample['length']          = length
            # print(sample)
            # import pdb; pdb.set_trace()
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        itype, device = self._itype_device
        
        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, itype)
                max_len = np.max(lengths) + 1
                offsets = np.random.randint(max_len - lengths)
                field_columns['offset'] = offsets
                tensor = lengths
            elif field == 'noise_type':
                noise_type = column
                continue
            else: # char & _validity
                size_diff = -1 if field == 'second_validity' else 0
                tensor = np.zeros([batch_size, max_len + size_diff], np.bool if field.endswith('_validity') else itype)
                for i, (values, offset, length) in enumerate(zip(column, offsets, lengths)):
                    tensor[i, offset:offset + length + size_diff] = values
            field_columns[field] = tensor

        for f, column in field_columns.items():
            field_columns[f] = torch.as_tensor(column, dtype = torch.bool if f.endswith('_validity') else torch.long, device = device)
        field_columns['noise_type'] = noise_type # no need to be on GPU
        return field_columns
