from os.path import join
from utils.pickle_io import pickle_load
from data.io import TreeSpecs, get_fasttext, encapsulate_vocabs
from data.delta import xtype_to_logits, logits_to_xtype
from collections import defaultdict, namedtuple
from utils.param_ops import HParams
from utils.file_io import parpath
from utils.types import NIL, UNK, BOS, EOS, M_TRAIN

BatchSpec = namedtuple('BatchSpec', 'size, iter')

class _BaseReader:
    def __init__(self,
                 vocab_dir,
                 i2vs, oovs,
                 paddings,
                 **to_model):
        self._vocab_dir = vocab_dir
        self._paddings = paddings
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        to_model['paddings'] = paddings
        self._to_model = {}
        self._oovs = oovs
        self.update(i2vs, **to_model)
        
    def dir_join(self, fname, pardir = 0):
        if parpath:
            fpath = parpath(self._vocab_dir, pardir)
        else:
            fpath = self._vocab_dir
        return join(fpath, fname)

    def update(self, i2vs, **to_model):
        i2vs, v2is = encapsulate_vocabs(i2vs, self._oovs)
        self._i2vs = HParams(i2vs)
        self._v2is = v2is
        if to_model:
            if self._to_model:
                for k,v in self._to_model.items():
                    if k not in to_model:
                        to_model[k] = v # perserve old values
            self._to_model = to_model
        else:
            to_model = self._to_model
        to_model.update({f'num_{k}s':v[0] for k,v in v2is.items()})

    def change_oovs(self, field, offset):
        if field in self._oovs:
            self._oovs[field] += offset

    @property
    def i2vs(self):
        return self._i2vs

    @property
    def v2is(self):
        return self._v2is

    @property
    def paddings(self):
        return self._paddings

    @property
    def device(self):
        return self._device

    def get_to_model(self, name):
        return self._to_model[name]

    def __str__(self):
        s = 'BaseReader Specs:\n'
        for f, v in self._i2vs._nested.items():
            s += f'  vocab of {f}: {len(v)} tokens with'
            if f in self._paddings:
                bos, eos = self._paddings[f]
                s += f' {v[bos]}({bos}) & {v[eos]}({eos})'
                if f == 'label':
                    bos, eos = self._paddings['xtype']
                    rox = logits_to_xtype(bos)
                    lox = logits_to_xtype(eos)
                    s += f' | {rox}({bos}) & {lox}({eos})\n'
                else:
                    s += '\n'
            else:
                s += f' {v[0]}(0)\n'
        return s

from utils.param_ops import change_key
class WordBaseReader(_BaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size,
                 load_nil,
                 i2vs, oovs):
        change_key(i2vs, 'word', 'token')
        self._info = pickle_load(join(vocab_dir, 'info.pkl'))
        weights = get_fasttext(join(vocab_dir, 'word.vec'))
        paddings = {}
        if vocab_size is None:
            if load_nil:
                weights[0] = 0
            else:
                i2v = i2vs['token'] = i2vs['token'][1:] + [BOS, EOS]
                weights = weights[1:]
                num = len(i2v)
                paddings['token'] = (num-2, num-1)
        else:
            assert vocab_size <= len(i2vs['token'])
            if load_nil:
                weights[0] = 0
                tokens = i2vs['token'][:vocab_size-1]
                tokens.append(UNK)
                weights = weights[:vocab_size-1]
            else:
                tokens = i2vs['token'][1:vocab_size-2] + [UNK, BOS, EOS]
                weights = weights[1:vocab_size-2] # space will be filled
                paddings['token'] = (vocab_size-2, vocab_size-1)
            assert len(tokens) == vocab_size, f'{len(tokens)} != {vocab_size}'
            i2vs['token'] = tokens
            oovs['token'] = vocab_size - 3

        if not load_nil:
            if 'tag' in i2vs:
                i2v = i2vs['tag'] = i2vs['tag'][1:] + [BOS, EOS]
                num = len(i2v)
                paddings['tag'] = (num-2, num-1)
            if 'label' in i2vs: # NIL is natural in labels
                i2v = i2vs['label'] = i2vs['label'] + [BOS, EOS]
                num = len(i2v)
                paddings['label'] = (num-2, num-1)
                paddings['xtype'] = (xtype_to_logits('>s', False), xtype_to_logits('<s', False))

        if 'label' not in i2vs:
            assert 'ftag'  not in i2vs

        if paddings:
            assert all(x in paddings for x in ('token', 'tag'))
            assert all(len(x) == 2 for x in paddings.values())
        super().__init__(vocab_dir, i2vs, oovs, paddings, initial_weights = weights)

    @property
    def info(self):
        return self._info

    def extend_vocab(self, extra_vocab, extra_weights):
        i2vs = self._i2vs._nested
        token = list(i2vs.pop('token'))
        ext_token = []
        ext_index = []

        # TODO: check diff vocab settings
        # checked: full+nil
        # import pdb; pdb.set_trace()
        for tid, tok in enumerate(extra_vocab):
            if tok not in token and tok not in (NIL, BOS, EOS, UNK):
                ext_index.append(tid)
                ext_token.append(tok)

        sep = len(token)
        while token[sep - 1] in (UNK, EOS, UNK):
            sep -= 1

        i2vs['token'] = token[:sep] + ext_token + token[sep:]
        weights = self.get_to_model('initial_weights')
        weights = np.concatenate([weights, extra_weights[ext_index]])
        self.change_oovs('token', len(ext_index))
        self.update(i2vs, initial_weights = weights)
        # import pdb; pdb.set_trace()
        

class SequenceBaseReader(_BaseReader):
    def __init__(self,
                 vocab_dir,
                 i2vs):
        i2vs, v2is = encapsulate_vocabs(i2vs, {})
        super().__init__(vocab_dir, i2vs, v2is, {})

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

E_MODE = 'plain', 'increase', 'bucket'
M_PLN, M_INC, M_BKT = E_MODE

def token_first(fdict):
    return (('token', fdict.pop('token')),) + tuple(fdict.items())

class LengthOrderedDataset(Dataset):
    def __init__(self,
                 heads,
                 lengths,
                 factors,
                 min_len,
                 max_len,
                 extra_text_helper):
        if min_len is None:
            min_len = 0
        if max_len is None:
            max_len = max(lengths)
        indices = defaultdict(list)
        for i, length in enumerate(lengths):
            if min_len <= length <= max_len:
                indices[length].append(i)

        self._heads = ('length',) + heads # create order
        self._indices = indices
        self._lengths = lengths
        self._mode = None
        self._extra_text_helper = extra_text_helper
        if factors:
            factors = tuple(factors.items())
            if len(factors) > 1:
                factors = tuple(zip(*factors))
            else:
                factors = factors[0][0]
        self._factors = factors # none, str or f-p

    @property
    def heads(self):
        return self._heads

    def plain_mode(self):
        plain_indices = []
        for length in sorted(self._indices):
            plain_indices.extend(self._indices[length])
        self._plain_indices = plain_indices
        self._mode = M_PLN

    def increasing_mode(self, tolerance = 2, avoid_randomness = False, self_reinit = True):
        to_sample = sorted(self._indices.keys())
        buffer = []
        
        self._inc_mode = to_sample, tolerance, buffer
        self._inc_avoid_randomness = avoid_randomness
        self._self_reinit = self_reinit
        self._inc_buffer_size = 0
        self._mode = M_INC
        self.__replenish_inc_buffer(append = False)

    def bucketed_mode(self, bucket_len, self_reinit = True):
        buckets = {}
        for l, idx in self._indices.items():
            group_id = l // bucket_len
            if group_id in buckets:
                buckets[group_id].extend(idx)
            else:
                buckets[group_id] = idx.copy()
        self._mode = M_BKT
        self._bkt_mode = bucket_len, buckets
        self._bkt_next_bucket = None
        self._self_reinit = self_reinit

    def __take_bkt_buffer(self, idx):
        bucket_len, buckets = self._bkt_mode
        if self._bkt_next_bucket is None:
            group_ids, bkt = zip(*buckets.items())
            bucket_probs = np.asarray([len(x) for x in bkt], dtype = np.float32)
            total = int(sum(bucket_probs))
            bucket_probs /= total
            group_id = np.random.choice(group_ids, p = bucket_probs)
            self._bkt_next_bucket = group_id
            self._bkt_buffer_size = total - 1
        else:
            group_id = self._bkt_next_bucket
            self._bkt_buffer_size -= 1
        bucket = buckets[group_id]
        idx = bucket.pop(idx % len(bucket))
        if len(bucket) == 0:
            buckets.pop(group_id)
            if buckets:
                self._bkt_next_bucket = min(buckets, key = lambda k: abs(group_id - k)) # find similar samples for batch
            else:
                self._bkt_next_bucket = None # final in a epoch
        return idx

    def __take_inc_buffer(self, idx):
        pointer = 0
        to_sample, _, buffer = self._inc_mode
        seg_size = len(buffer[pointer])
        while seg_size <= idx:
            # clean buffer through the buffer
            if seg_size == 0:
                buffer.pop(pointer)
                to_sample.pop(pointer)
                continue
            # locate pointer
            pointer += 1
            idx -= seg_size
            seg_size = len(buffer[pointer])
        self._inc_buffer_size -= 1
        if seg_size == 1: # last chunk
            idx = buffer.pop(pointer).pop(0)
            to_sample.pop(pointer)
            if pointer == 0:
                self.__replenish_inc_buffer(append = True)
        else:
            idx = buffer[pointer].pop(idx)
        return idx

    def __replenish_inc_buffer(self, append):
        to_sample, tolerance, buffer = self._inc_mode
        if len(to_sample) == 0:
            return False
        if append:
            pointer = len(buffer)
            if pointer >= len(to_sample):
                return False
        else:
            pointer = 0
        min_len = to_sample[0]

        while to_sample[pointer] <= min_len + tolerance:
            seg = self._indices[to_sample[pointer]].copy()
            buffer.append(seg)
            self._inc_buffer_size += len(seg)
            pointer += 1
            if pointer == len(to_sample):
                return False # end of the tape
        return True
            
    def __len__(self):
        return sum(len(s) for s in self._indices.values())
        
    def __getitem__(self, idx):

        factor = self._factors
        if isinstance(factor, tuple): # or is None or str
            factors, probs = factor
            factor = np.random.choice(factors, p = probs)
            # print(factor)

        if self._mode == M_PLN:
            idx = self._plain_indices[idx]
        elif self._mode == M_INC:
            idx = 0 if self._inc_avoid_randomness else (idx % self._inc_buffer_size)
            idx = self.__take_inc_buffer(idx)
        elif self._mode == M_BKT:
            idx = self.__take_bkt_buffer(idx)

        length = self._lengths[idx]
        sample = self.at_idx(idx, factor, length)
        sample = tuple(sample[h] for h in self._heads)
        if self._extra_text_helper is not None:
            self._extra_text_helper.buffer(idx)
        
        return sample


    def at_idx(self, idx, factor, length):
        raise NotImplementedError()

    def _collate_fn(self, batch):
        raise NotImplementedError()

    def collate_fn(self, batch):
        field_columns = self._collate_fn(batch)
        if self._extra_text_helper:
            field_columns.update(self._extra_text_helper.get())

        # internal preparation
        if self._mode == M_INC and self._self_reinit and self._inc_buffer_size == 0:
            to_sample = sorted(self._indices.keys())
            self._inc_mode = (to_sample,) + self._inc_mode[1:]
            self.__replenish_inc_buffer(append = False)
        elif self._mode == M_BKT:
            if self._self_reinit and self._bkt_buffer_size == 0:
                bucket_len, _ = self._bkt_mode
                self.bucketed_mode(bucket_len)
                # print('END N END', flush = True)
            else:
                self._bkt_next_bucket = None

        return field_columns


def post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size):
    if mode != M_TRAIN:
        len_sort_ds.plain_mode()
    elif sort_by_length:
        if bucket_length > 0:
            len_sort_ds.increasing_mode(bucket_length)
        else:
            len_sort_ds.plain_mode()
    else:
        len_sort_ds.bucketed_mode(bucket_length)
    di = DataLoader(len_sort_ds, batch_size = batch_size, collate_fn = len_sort_ds.collate_fn, shuffle = mode == M_TRAIN)#, num_workers = 1) # no way to get more!
    return BatchSpec(len(len_sort_ds), di)