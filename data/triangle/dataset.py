from data.backend import LengthOrderedDataset, np, torch, token_first
from data.delta import s_index
from utils.file_io import read_data
from tqdm import tqdm
from time import time
from data.delta import write_tensors, E_XDIM

fields = 'token', 'tag', 'ftag'
fieldx = 'label', 'xtype'
# FieldOrder = 'token', 'tag', 'label', 'xtype', 'ftag', 'length'

class TriangularDataset(LengthOrderedDataset):
    def __init__(self,
                 dir_join,
                 prefix,
                 field_v2is,
                 paddings,
                 device,
                 factors  = None,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        heads   = []
        columns = {}
        if 'label' in field_v2is or 'polar' in field_v2is:
            field_v2is = field_v2is.copy()
            field_v2is['xtype'] = (len(E_XDIM), int)

        for field, v2i in token_first(field_v2is):
            start = time()
            heads.append(field)
            if field in fields: # token /tag / ftag / finc
                print(f'Load Dataset {prefix}.{field}', end = ' ', flush = True)
                if field == 'token':
                    column, lengths, text = read_data(dir_join(f'{prefix}.word'), v2i, True)
                else:
                    column = read_data(dir_join(f'{prefix}.{field}'), v2i, False)
                columns[field] = column
            elif factors is None:
                with tqdm(total = len(lengths), desc = f'Load Dataset {prefix}.{field}') as qbar:
                    columns[field] = read_data(dir_join(f'{prefix}.{field}'), v2i, qbar = qbar)
            else:
                for factor, prob in factors.items():
                    with tqdm(total = len(lengths), desc = f'Load Dataset {prefix}.{field}.{factor}') as qbar:
                        column = read_data(dir_join(f'{prefix}.{field}.{factor}'), v2i, qbar = qbar)
                    columns[(field, factor)] = column
            print(f'in {time() - start:.2f}s')
        assert all(len(lengths) == len(col) for col in columns.values())
        
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, device)
        heads = tuple(heads)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)

        self._columns = columns
        self._paddings_device = paddings, device

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
        dtype = np.int32
        field_columns = {}
        paddings, device = self._paddings_device

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
