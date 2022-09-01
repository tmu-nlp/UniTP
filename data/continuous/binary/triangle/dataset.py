from data.backend import LengthOrderedDataset, np, torch, token_first
from data.binary import s_index
from utils.file_io import read_data
from utils.types import device
from tqdm import tqdm
from data.binary import write_tensors, E_XDIM

fields = 'token', 'tag', 'ftag'
fieldx = 'label', 'xtype'
# FieldOrder = 'token', 'tag', 'label', 'xtype', 'ftag', 'length'

class TriangularDataset(LengthOrderedDataset):
    def __init__(self,
                 dir_join,
                 prefix,
                 field_v2is,
                 paddings,
                 factors  = None,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        heads   = []
        columns = {}
        has_char = 'char' in field_v2is
        has_xtype = 'label' in field_v2is or 'polar' in field_v2is
        if has_char or has_xtype:
            field_v2is = field_v2is.copy()
            if has_char:
                _, c2i = field_v2is.pop('char')
            if has_xtype:
                field_v2is['xtype'] = (len(E_XDIM), int)
        field_v2is = token_first(field_v2is)
        num_fields_left = -1
        field_names = []
        for f, _ in field_v2is:
            if f in fields:
                field_names.append(f)
                num_fields_left += 1
            else:
                num_factors = len(factors) if factors else 1
                field_names.append(f + f'({num_factors})')
                num_fields_left += num_factors
        
        with tqdm(desc = 'Load Datasets: ' + ', '.join(field_names)) as qbar:
            for field, v2i in field_v2is:
                qbar.desc = 'Load Datasets: \033[32m' + ', '.join((f + '\033[m') if f.startswith(field) else f for f in field_names)
                if field in fields: # token /tag / ftag / finc
                    if field == 'token':
                        column, lengths, text = read_data(dir_join(f'{prefix}.word'), v2i, True)
                        qbar.total = num_fields_left * len(lengths)
                    else:
                        column = read_data(dir_join(f'{prefix}.{field}'), v2i, qbar = qbar)
                    columns[field] = column
                elif factors is None:
                    columns[field] = read_data(dir_join(f'{prefix}.{field}'), v2i, qbar = qbar)
                else:
                    for factor in factors:
                        columns[(field, factor)] = read_data(dir_join(f'{prefix}.{field}.{factor}'), v2i, qbar = qbar)
                heads.append(field)
        assert all(len(lengths) == len(col) for col in columns.values())
        
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, c2i if has_char else None)
        heads = tuple(heads)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)

        self._columns = columns
        self._paddings = paddings

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = {}
        for field, column in self._columns.items():
            if isinstance(field, tuple) and field[1] == factor:
                sample[field[0]] = column[idx]
            else:
                sample[field]    = column[idx]
        sample['length'] = length
        return sample

    def _collate_fn(self, batch):
        dtype = np.int32
        field_columns = {}
        paddings = self._paddings

        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, dtype)
                max_len = np.max(lengths)
                if paddings:
                    max_len += 2 # BOS and EOS
                tri_len = s_index(max_len)
                offsets = (max_len - lengths) // 2
                field_columns['offset'] = offsets
                tensor = lengths
            elif field in fields: # token or tags
                tensor = np.zeros([batch_size, max_len], dtype)
                for i, (values, offset, length) in enumerate(zip(column, offsets, lengths)):
                    end = offset + length
                    tensor[i, offset:end] = values
                    if paddings:
                        bid, eid = paddings[field]
                        tensor[i, :offset] = bid
                        tensor[i, end:]    = eid
            else:
                tensor = column, np.zeros([batch_size, tri_len], dtype)
            field_columns[field] = tensor

        if 'xtype' in field_columns: # 'label' or 'polar' in field_columns and 
            _label_ = 'label' if 'label' in field_columns else 'polar'
            paddings = paddings[_label_] + paddings['xtype'] if paddings else None
            label, label_tensor = field_columns[_label_]
            xtype, xtype_tensor = field_columns['xtype']
            for offset, l, lt, x, xt in zip(offsets, label, label_tensor, xtype, xtype_tensor):
                write_tensors(l, x, lt, xt, offset, paddings)
            field_columns[_label_] = label_tensor
            field_columns['xtype'] = xtype_tensor
            if _label_ == 'label':
                field_columns['top3_label'] = np.stack([np.asarray(x[:3], dtype) for x in label])

        for f, column in field_columns.items():
            field_columns[f] = torch.as_tensor(column, dtype = (None if f == 'xtype' else torch.long), device = device)
        return field_columns

from data.mp import DM
from data.triangle import triangle_to_layers, after_to_tree
class TriangularDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, offsets, lengths, token, tag, label, right):
        for offset, length, tokens, tags, labels, rights in zip(offsets, lengths, token, tag, label, right):
            size = len(tokens)
            token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
            tag_layer   = tuple(i2t[t] for t in tags  [offset:offset+length])
            label_layers = triangle_to_layers(labels, size, offset, length, i2l)
            right_layers = triangle_to_layers(rights, size, offset, length, None)
            tree = after_to_tree(token_layer, tag_layer, label_layers, right_layers)
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        start = seg_id * seg_size
        if start < batch_size:
            return tuple(x[start: (seg_id + 1) * seg_size] for x in args)