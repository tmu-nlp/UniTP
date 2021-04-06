from data.backend import WordBaseReader, LengthOrderedDataset, post_batch
from data.ner_types import M_TRAIN, M_DEVEL, M_TEST, NIL, split_files, read_dataset, remove_bio_prefix
from data.io import load_i2vs, join
from utils.types import vocab_size, train_batch_size, train_max_len, train_bucket_len, true_type, false_type
import numpy as np
import torch

split_files = {v:k + '.txt' for k, v in split_files.items()}

data_config = dict(vocab_size     = vocab_size,
                   batch_size     = train_batch_size,
                   with_bi_prefix = true_type,
                   with_pos_tag   = false_type,
                   max_len        = train_max_len,
                   bucket_len     = train_bucket_len,
                   sort_by_length = false_type)

class NerReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 corpus_path,
                 with_bi_prefix,
                 with_pos_tag,
                 vocab_size = None,
                 extra_text_helper = None):
        vocabs = '' if extra_text_helper is None else 'char '
        if with_bi_prefix:
            vocabs += 'word bio'
        else:
            vocabs += 'word ner'
        if with_pos_tag:
            vocabs += ' pos'
        i2vs = load_i2vs(vocab_dir, vocabs.split())
        super().__init__(vocab_dir, vocab_size, True, i2vs, {})
        self._load_options = corpus_path, extra_text_helper

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              o_split_rate   = 0,
              o_split_whole  = False,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        assert mode in (M_TRAIN, M_DEVEL, M_TEST)
        corpus_path, extra_text_helper = self._load_options

        len_sort_ds = NerDataset(join(corpus_path, split_files[mode]),
                                 o_split_rate,
                                 o_split_whole,
                                 self.v2is,
                                 self.device,
                                 min_len,
                                 max_len, 
                                 extra_text_helper)

        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)

from random import random
class NerDataset(LengthOrderedDataset):
    def __init__(self,
                 fname,
                 o_split_rate,
                 o_split_whole,
                 v2is,
                 device,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        lengths, text = [], []
        token, pos = [], []
        p2is = n2is = b2is = None
        (_, t2is) = v2is['token']
        if 'pos' in v2is:
            (_, p2is) = v2is['pos']
        if 'bio' in v2is:
            _, b2is = v2is['bio']
            bio = []
        else:
            _, n2is = v2is['ner']
            ner, ner_fence = [], []
        for words, pos_tags, bio_tags in read_dataset(fname):
            lengths.append(len(words))
            if extra_text_helper:
                text.append(words)
            token.append([t2is(w) for w in words])
            if p2is:
                pos.append([p2is(p) for p in pos_tags])
            if b2is:
                bio.append([b2is(x) for x in bio_tags])
            else:
                _ner_fence, _ner = remove_bio_prefix(bio_tags)
                ner.append([n2is(x) for x in _ner])
                ner_fence.append(_ner_fence)

        if b2is:
            heads = ('bio',)
            columns = dict(token = token, bio = bio)
            factors = None
        else:
            heads = ('ner', 'fence')
            columns = dict(token = token, ner = ner, fence = ner_fence)
            self._o_split_whole_idx = o_split_rate, o_split_whole, n2is('O')
            factors = {False: 1 - o_split_rate, True: o_split_rate} if o_split_whole else o_split_rate > 0
        if p2is:
            heads = ('token', 'pos') + heads
            columns['pos'] = pos
        else:
            heads = ('token',) + heads
        assert all(len(lengths) == len(col) for col in columns.values())

        if extra_text_helper:
            _, c2is = v2is['char']
            extra_text_helper = extra_text_helper(text, device, c2is)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)
        self._columns = columns
        self._device = device

    def at_idx(self, idx, factor, length):
        sample = {}
        for hd in self.heads:
            if hd == 'length':
                value = length
            else:
                value = self._columns[hd][idx]
                if factor and hd == 'fence': # 0359 -> 012356789
                    old_ner = sample['ner']  #  OPO ->  OOOPOOOO
                    new_ner, new_fence = [], [0]
                    o_split_rate, o_split_whole, o_idx = self._o_split_whole_idx
                    for start, end, ner in zip(value, value[1:], old_ner):
                        if ner == o_idx and end - start > 1:
                            new_range = range(start + 1, end + 1)
                            if not o_split_whole:
                                new_range = [i for i in new_range if i == end or random() < o_split_rate]
                            new_ner.extend(ner for _ in new_range)
                            new_fence.extend(new_range)
                        else:
                            new_ner.append(ner)
                            new_fence.append(end)
                    sample['ner'] = new_ner
                    value = new_fence
            sample[hd] = value
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                tensor = lengths = np.asarray(column, np.uint8)
                max_len = lengths.max()
            elif field in ('token', 'pos', 'bio'):
                tensor = np.zeros([batch_size, max_len], np.int32 if field == 'token' else np.uint8)
                for i, (values, length) in enumerate(zip(column, lengths)):
                    tensor[i, :length] = values
            elif field == 'ner':
                ner_lengths = np.array([len(x) for x in column], np.uint8)
                max_ner_len = ner_lengths.max()
                tensor = np.zeros([batch_size, max_ner_len], np.uint8)
                for i, (values, length) in enumerate(zip(column, ner_lengths)):
                    tensor[i, :length] = values
            else:
                assert field == 'fence'
                tensor = np.zeros([batch_size, max_len + 1], np.bool)
                for i, indices in enumerate(column):
                    tensor[i, indices] = True
            dtype = torch.bool if field == 'fence' else torch.long
            field_columns[field] = torch.as_tensor(tensor, dtype = dtype, device = self._device)
        return field_columns