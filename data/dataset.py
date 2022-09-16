import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple
from itertools import zip_longest

BatchSpec = namedtuple('BatchSpec', 'size, iter')

E_MODE = 'plain', 'increase', 'bucket'
M_PLN, M_INC, M_BKT = E_MODE

def token_first(fdict):
    return (('token', fdict.pop('token')),) + tuple(fdict.items())

def seg_size(layers):
    return [len(l) for l in layers]


def pad_layer_nil(layer, length):
    return layer + [0] * (length - len(layer))

def pad_layer_bos_eos(layer, length, offset, bos, eos):
    return [bos] * offset + layer + [eos] * (length - len(layer) + 1)

def pad_layers_nil(layers, segments):
    base = []
    for layer, length in zip(layers, segments):
        base += pad_layer_nil(layer, length)
    for length in segments[len(layers):]:
        base += pad_layer_nil(layer, length)
    return base

def pad_layers_bos_eos(layers, offset, segments, bos, eos, eos_):
    base = []
    for layer, length in zip(layers, segments):
        base += pad_layer_bos_eos(layer, length, offset, bos, eos)
    if eos != eos_ and len(layer) < len(segments):
        count = 0
        while base[-1] == eos:
            count += base.pop() == eos
        base += [eos_] * count
    for length in segments[len(layers):]:
        base += pad_layer_bos_eos(layer, length, offset, bos, eos_)
    return base

def pad_tag_like_nil(batch, length, pool = None):
    if pool is not None:
        return pool.starmap(pad_layer_nil, ((layers, length) for layers in batch))
    return [pad_layer_nil(layers, length) for layers in batch]

def pad_tag_like_bos_eos(batch, length, offsets, bos, eos, pool = None):
    if pool is not None:
        return pool.starmap(pad_layer_bos_eos, (args + (length, bos, eos) for args in zip(batch, offsets)))
    return [pad_layer_bos_eos(*args, length, bos, eos) for args in zip(batch, offsets)]


def pad_label_like_nil(batch, segments, pool = None):
    if pool is not None:
        return pool.starmap(pad_layers_nil, ((layers, segments) for layers in batch))
    return [pad_layers_nil(layers, segments) for layers in batch]

def pad_label_like_bos_eos(batch, segments, offsets, bos, eos, eos_, pool = None):
    if pool is not None:
        return pool.starmap(pad_layers_bos_eos, (args + (segments, bos, eos, eos_) for args in zip(batch, offsets)))
    return [pad_layers_bos_eos(*args, segments, bos, eos, eos_) for args in zip(batch, offsets)]


def erect_joint_layers(bid, lengths, layers, segments, offset, boundary):
    cumu = 0
    base = []
    for l_len, joint, s_len in zip(lengths, layers, segments):
        base.extend((bid, cumu + j + offset) for j in joint) # 3:4:0__3_ _12_4
        base.extend((bid, cumu + j + offset) for j in range(l_len + 1, s_len + 1))
        cumu += s_len + 1
    if boundary is None:
        boundary = lengths[-1] + 1
    for s_len in (segments[len(layers):]):
        base.extend((bid, cumu + c + offset) for c in range(boundary, s_len + 1)) # 1:01_ __2
        cumu += s_len + 1
    return base

def erect_split_layers(bid, layers, segments, offset):
    cumu = 0
    base = []
    for split, s_len in zip(layers, segments):
        base.extend((bid, cumu + s + offset) for s in split)
        cumu += s_len + 1
    for s_len in segments[len(layers):]:
        base.extend((bid, cumu + s + offset) for s in (0, 1))
        cumu += s_len + 1
    return base

def erect_joint_like(lengths, batch, segments, offset):
    joint = []
    segments, bnd = ([segments], None) if isinstance(segments, int) else (segments, 2)
    for eid, joint_layers in enumerate(zip(lengths, batch)):
        joint += erect_joint_layers(eid, *joint_layers, segments, offset, bnd)
    return joint

def erect_split_like(batch, segments, offset):
    split = []
    for eid, layers in enumerate(batch):
        split += erect_split_layers(eid, layers, segments, offset)
    return split


class LengthOrderedDataset(Dataset):

    def __init__(self,
                 heads,
                 seg_head,
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

        self._heads = namedtuple('Batch', heads) # create order
        self._indices = indices
        self._lengths = lengths
        self._segid = heads.index(seg_head) if seg_head else None
        self._mode = None
        self._extra_text_helper = extra_text_helper
        self._reset_factors(factors)
        from utils.types import device
        LengthOrderedDataset.device = device

    def _reset_factors(self, factors):
        if isinstance(factors, dict):
            total = 0
            factor = None
            for k, v in factors.items():
                if v == 1:
                    factor = k
                total += v
            assert total == 1
            if factor is None:
                factors = tuple((k, v) for k, v in factors.items() if v)
                if len(factors) > 1:
                    factors = tuple(zip(*factors))
                else:
                    factors = factors[0][0]
            else:
                factors = factor
        self._factors = factors # none, str or f-p

    @property
    def heads(self):
        return self._heads._fields

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
            
    def __len__(self): # pytorch need this
        return sum(len(s) for s in self._indices.values())

    @property
    def size(self): # for our data augmentation
        return self.__len__()
        
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

        helper_outputs = None
        if self._extra_text_helper is not None:
            helper_outputs = self._extra_text_helper.buffer(idx)
        sample = self.at_idx(idx, factor, helper_outputs)
        length = self._lengths[idx]
        if (fid := self._segid) is None:
            return (length,) + sample
        return (length, seg_size(sample[fid])) + sample

    def at_idx(self, idx, factor, helper_outputs):
        raise NotImplementedError()

    def _collate_fn(self, batch, length, segment):
        raise NotImplementedError()

    def collate_fn(self, batch):
        batch = tuple(zip(*batch))
        length = np.asarray(batch[0])
        if self._segid is None:
            segment = None
            batch = self._heads(*batch[1:])
        else:
            segment = np.asarray([l for l in zip_longest(*batch[1], fillvalue = 0)]).transpose()
            batch = self._heads(*batch[2:])

        field_columns = self._collate_fn(batch, length, segment)
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


from utils.types import device, M_TRAIN
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
    di = DataLoader(len_sort_ds,
                    batch_size = batch_size,
                    collate_fn = len_sort_ds.collate_fn,
                    shuffle = mode == M_TRAIN)
                    # num_workers = 1) # no way to get more!
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    return BatchSpec(len_sort_ds.size, di)