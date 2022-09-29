import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, namedtuple
from itertools import zip_longest
from utils.types import is_roughly_zero, F_CNF, F_CON

NewBin = namedtuple('NewBin', 'sentence_level, left, right, msub')
ConBin = namedtuple('ConBin', 'cache, con')
DivBin = namedtuple('DivBin', 'cache, div')

BatchSpec = namedtuple('BatchSpec', 'size, iter')

E_MODE = 'plain', 'increase', 'bucket'
M_PLN, M_INC, M_BKT = E_MODE

def token_first(fdict):
    return (('token', fdict.pop('token')),) + tuple(fdict.items())

def seg_size(layers):
    return [len(l) for l in layers]


nil_pad = [0]
def pad_layer_nil(layer, length, pre = 0):
    return pre * nil_pad + layer + nil_pad * (length - len(layer))

def pad_layer_bos_eos(layer, length, offset, bos, eos):
    return [bos] * offset + layer + [eos] * (length - len(layer) + 1)

def pad_layers_nil(layers, segments, pre = 0, final_layer = None):
    base = []
    for layer, length in zip(layers, segments):
        base += pad_layer_nil(layer, length, pre)
    if final_layer is not None:
        layer = final_layer
    for length in segments[len(layers):]:
        base += pad_layer_nil(layer, length, pre)
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

def pad_tag_like_nil(batch, length, pre = 0, pool = None):
    if pool is not None:
        return pool.starmap(pad_layer_nil, ((layers, length, pre) for layers in batch))
    return [pad_layer_nil(layers, length, pre) for layers in batch]

def pad_tag_like_bos_eos(batch, length, offsets, bos, eos, pool = None):
    if pool is not None:
        return pool.starmap(pad_layer_bos_eos, (args + (length, bos, eos) for args in zip(batch, offsets)))
    return [pad_layer_bos_eos(*args, length, bos, eos) for args in zip(batch, offsets)]


def pad_label_like_nil(batch, segments, pre = 0, final_layer = None, pool = None):
    if pool is not None:
        return pool.starmap(pad_layers_nil, ((layers, segments, pre, final_layer) for layers in batch))
    return [pad_layers_nil(layers, segments, pre, final_layer) for layers in batch]

def pad_label_like_bos_eos(batch, segments, offsets, bos, eos, eos_, pool = None):
    if pool is not None:
        return pool.starmap(pad_layers_bos_eos, (args + (segments, bos, eos, eos_) for args in zip(batch, offsets)))
    return [pad_layers_bos_eos(*args, segments, bos, eos, eos_) for args in zip(batch, offsets)]


def erect_joint_layers_more(bid, lengths, layers, segments, offset, boundary):
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

def erect_split_layers_more(bid, layers, segments, offset):
    cumu = 0
    base = []
    for split, s_len in zip(layers, segments):
        base.extend((bid, cumu + s + offset) for s in split)
        cumu += s_len + 1
    for s_len in segments[len(layers):]:
        base.extend((bid, cumu + s + offset) for s in (0, 1))
        cumu += s_len + 1
    return base

def erect_joint_more(lengths, batch, segments, offset):
    joint = []
    segments, bnd = ([segments], None) if isinstance(segments, int) else (segments, 2)
    for eid, joint_layers in enumerate(zip(lengths, batch)):
        joint += erect_joint_layers_more(eid, *joint_layers, segments, offset, bnd)
    return joint

def erect_split_more(batch, segments, offset):
    split = []
    for eid, layers in enumerate(batch):
        split += erect_split_layers_more(eid, layers, segments, offset)
    return split


def erect_layers(bid, layers, segments, offset):
    cumu = 0
    base = []
    for joint, s_len in zip(layers, segments):
        base.extend((bid, cumu + j + offset) for j in joint)
        cumu += s_len
    return base

def erect_joint_less(batch, segments, offset):
    base = []
    for eid, joint in enumerate(batch):
        base += erect_layers(eid, joint, segments, offset)
    return base

def fill_bool_tensor(idx, tensor, value, indice_args):
    if idx:
        idx = torch.as_tensor(idx, **indice_args)
        tensor[idx[:, 0], idx[:, 1]] = value

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
        from data.cross.dataset import InterLayerDisco
        InterLayerDisco.tensor_args.update(device = device)

    def _reset_factors(self, factors):
        if isinstance(factors, dict):
            total = 0
            factor = None
            for k, v in factors.items():
                if v == 1:
                    factor = k
                total += v
            assert is_roughly_zero(total - 1)
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

    def reset_binary_factor(self, factor, esub, msub, *, initialize = None):
        level, left, right = factor
        esub_factor = factor = {0: 1 - esub, 1: esub}
        sentence_level = level == 'sentence'
        # sentence-cnf-num: old a little dynamic; sentence/phrase-continuous-num: old static
        # phrase-cnf-num or old but msub > 0; sentence/phrase-num-num: beta(num, num)
        if old_cnf := (sentence_level and left in (F_CNF, F_CON) and msub == 0):
            cache_n = 2
            if left == F_CON:
                if isinstance(right, int):
                    cache_n = right
                    prob_n = (1 / cache_n,) * cache_n
                elif isinstance(right, tuple):
                    cache_n = len(right)
                    prob_n = right
            if initialize:
                cache = tuple([None] * (cache_n << 1) for _ in range(initialize))
            if left == F_CON and isinstance(right, float):
                extra = lambda cache: ConBin(cache, right)
            else:
                factor = {} # overwrite
                if left == F_CON:
                    for i, prob in enumerate(prob_n):
                        factor[(1 << i)] = prob * (1 - esub)
                        factor[(1 << i) + (1 << cache_n)] = prob * esub
                    extra = lambda cache: DivBin(cache, cache_n)
                else:
                    for k, v in esub_factor.items():
                        factor[0 + (k << 1)] = (1 - right) * v
                        factor[1 + (k << 1)] = right * v
                    extra = cache
        else:
            extra = NewBin(sentence_level, left, right, msub)
        if initialize:
            return factor, (extra(cache) if callable(extra) else extra)

        if old_cnf:
            if isinstance(old_extra := self._args[-1], NewBin):
                cache = tuple([None] * (cache_n << 1) for _ in range(len(self._args[0])))
            else: # reuse old cache
                cache = old_extra[1] if isinstance(old_extra, (ConBin, DivBin)) else old_extra
            new_extra = extra(cache) if callable(extra) else cache
        else:
            new_extra = extra
        self._args = self._args[:-1] + (new_extra,)
        super()._reset_factors(factor)


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

def checkin_cache(cache, idx, data, the_same = None):
    for head in enumerate(cache):
        if head and ((head == data) if the_same is None else the_same(head, data)):
            cache[idx] = head
            break
    if cache[idx] is None:
        cache[idx] = data

def binary_signals(factor, idx, extra, signal_fn, the_same = None):
    if isinstance(extra, NewBin):
        sentence_level, left, right, msub = extra
        if isinstance(left, float):
            frac = np.random.beta(left, right) if sentence_level else (left, right)
        elif left == F_CNF:
            frac = (np.random.random() < right) if sentence_level else (F_CNF, right)
        else:
            assert left == F_CON and sentence_level and msub > 0
            frac = right
        signals = signal_fn(frac, factor, msub)
    else:
        if isinstance(extra, ConBin):
            cache, frac = extra
            esub = factor
        elif isinstance(extra, DivBin):
            cache, cache_n = extra
            esub = factor & (1 << cache_n)
            factor = int(factor & (esub - 1)).bit_length()
            frac = factor / (cache_n - 1)
        else:
            cache = extra
            frac = factor & 1
            esub = factor & 2
        cache = cache[idx]
        if cache[factor] is None:
            checkin_cache(cache, factor, signal_fn(frac, esub), the_same)
        signals = cache[factor]
    return signals


from data.mp import Process, mp_while
from utils.types import F_RANDOM
class FileWorker(Process):
    estimate_total = True

    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        i, q, fileids, reader, from_corpus, serialize_esub = self._args
        error_cnt = inst_cnt = n_proceed = l_estimate = 0
        n_fileid = len(fileids)

        for eid, fn in enumerate(fileids):
            trees = reader.parsed_sents(fn)
            n_proceed += len(trees)
            n_estimate = int(n_proceed * n_fileid / (eid + 1))

            for tree in trees:
                q.put((i, n_estimate) if n_estimate != l_estimate else i)
                try:
                    dx = from_corpus(tree)
                    dx.binary(F_RANDOM)
                    dx.multib()
                except:
                    error_cnt += 1
                    continue
            
                q.put((None, dx.serialize(serialize_esub)))
                inst_cnt += 1
        q.put((i, inst_cnt, error_cnt))

class SampleWorker(Process):
    estimate_total = False

    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        i, q, fileids, from_corpus, serialize_esub = self._args
        error_cnt = inst_cnt = 0

        for graph in fileids:
            q.put(i)
            try:
                dx = from_corpus(graph)
                dx.binary(F_RANDOM)
                dx.multib()
            except:
                error_cnt += 1
                continue

            q.put((None, dx.serialize(serialize_esub)))
            inst_cnt += 1
        q.put((i, inst_cnt, error_cnt))

def read_signals(w2i, t2i, fileids, reader, Signal, from_fn, esub, char_as_token = False):
    token, tag, signals, text = [], [], [], []
    def receive(t, qbar):
        if isinstance(t, int):
            qbar.update(t)
        elif len(t) == 2:
            tid, n_estimate_or_data = t
            if isinstance(tid, int):
                qbar.update(tid, total = n_estimate_or_data)
                qbar.update(tid)
            else:
                signal = Signal.instantiate(n_estimate_or_data)
                token  .append(signal.char_to_idx(w2i) if char_as_token else signal.word_to_idx(w2i))
                tag    .append(signal. tag_to_idx(t2i))
                text   .append(signal.word)
                signals.append(signal)
        else:
            return t
    if reader is None:
        mp_while(SampleWorker, fileids, receive, from_fn, esub > 0)
    else:
        mp_while(FileWorker, fileids, receive, reader, from_fn, esub > 0)
    return [len(wd) for wd in text], token, tag, signals, text