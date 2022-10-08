from data.vocab import VocabKeeper
from data.dataset import LengthOrderedDataset, post_batch
from data.ner_types import read_dataset, remove_bio_prefix
from data.io import load_i2vs, join, get_fasttext
import numpy as np
from tqdm import tqdm
import torch

def add_char_from_word(i2vs):
    chars = set()
    for word in i2vs['word'][1:]:
        chars.update(word)
    i2vs['char'] = [NIL, PAD] + sorted(chars)

from data import NIL, PAD
from data.utils import CharTextHelper
from utils.param_ops import change_key
class NerReader(VocabKeeper):
    def __init__(self, ner,
                 with_bi_prefix,
                 with_pos_tag,
                 extra_text_helper = None):
        if with_bi_prefix:
            vocabs = ['word', 'bio']
        else:
            vocabs = ['word', 'ner']
        if with_pos_tag:
            vocabs.append('pos')
        i2vs = load_i2vs(ner.local_path, *vocabs)
        change_key(i2vs, 'word', 'token')
        vocabs[0] = 'token'
        def token_fn(token_type):
            weights = get_fasttext(join('token', token_type + '.vec'))
            weights[0] = 0 # [nil ...]
            return weights
        char_vocab = None
        # if extra_text_helper is CharTextHelper:
        if extra_text_helper is CharTextHelper:
            add_char_from_word(i2vs)
            char_vocab = {}
            _, t2is = self.v2is['token']
            _, c2is = self.v2is['char']
            pad_idx = c2is(PAD)
            for word in tqdm(self.i2vs.token[1:], 'NER-Wordlist'): # skip pad <nil>
                char_vocab[t2is(word)] = [c2is(t) for t in word] + [pad_idx]
        self._load_options = ner, extra_text_helper, char_vocab
        super().__init__(vocabs, i2vs, {}, {}, weight_fn = token_fn)

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              min_len        = 0,
              max_len        = None,
              sort_by_length = True):
        ner, extra_text_helper, char_vocab = self._load_options

        len_sort_ds = NerDataset(join(ner.source_path, ner.build_params._nested[mode + '_set']),
                                 ner.ner_extension,
                                 char_vocab,
                                 self.v2is,
                                 self.device,
                                 min_len,
                                 max_len, 
                                 extra_text_helper)

        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)

from data.noise import insert_word, substitute_word, drop_word
from data.ner_types import insert_o, substitute_o, delete_o
from random import random, sample, randint
class NegativeAugment:
    def __init__(self, lengths, ner_extension, char_vocab, o_idx):
        shape = []
        aug_size = 0
        for length in lengths:
            insert_n = round(length * ner_extension.insert.a)
            delete_n = round(length * ner_extension.delete.a)
            substitute_n = round(length * ner_extension.substitute.a)
            insert_n = max(1, insert_n) if ner_extension.insert.a > 0 else 0
            delete_n = max(1, delete_n) if ner_extension.delete.a > 0 else 0
            substitute_n = max(1, substitute_n) if ner_extension.substitute.a > 0 else 0
            shape.append((length, insert_n, delete_n, substitute_n))
            aug_size += insert_n + delete_n + substitute_n
        self._aug_size = aug_size
        self._ner = ner_extension, o_idx, char_vocab, shape

    def __len__(self):
        return self._aug_size

    def insert(self, idx, token, ner, fence, sub_idx_fence):
        ner_extension, o_idx, char_vocab, shape = self._ner
        length, insert_n, _, _ = shape[idx]
        for nid in range(insert_n):
            f_indices = [i for i in range(length + 1) if random() < ner_extension.insert.p]
            if not f_indices: f_indices.append(randint(0, length))
            new_ner, new_fence = insert_o(ner, fence, f_indices, o_idx)
            values = sample(char_vocab.keys(), len(f_indices))
            new_token = insert_word(token, f_indices, values)
            if sub_idx_fence:
                sub_idx, sub_fence = sub_idx_fence
                new_sub_idx_fence = insert_word(sub_idx, f_indices, [char_vocab[i] for i in values], sub_fence)
                yield (len(new_token), new_token, new_ner, new_fence), new_sub_idx_fence
            else:
                yield (len(new_token), new_token, new_ner, new_fence), None

    def delete(self, idx, token, ner, fence, sub_idx_fence):
        ner_extension, _, _, shape = self._ner
        length, _, delete_n, _ = shape[idx]
        for nid in range(delete_n):
            n_indices = [i for i in range(length) if random() < ner_extension.delete.p]
            if not n_indices: n_indices.append(randint(0, length - 1))
            new_ner, new_fence = delete_o(ner, fence, n_indices)
            new_token = drop_word(token, n_indices)
            if sub_idx_fence:
                sub_idx, sub_fence = sub_idx_fence
                new_sub_idx_fence = drop_word(sub_idx, n_indices, sub_fence)
                yield (len(new_token), new_token, new_ner, new_fence), new_sub_idx_fence
            else:
                yield (len(new_token), new_token, new_ner, new_fence), None

    def substitute(self, idx, token, ner, fence, sub_idx_fence):
        ner_extension, o_idx, char_vocab, shape = self._ner
        length, _, _, substitute_n = shape[idx]
        for nid in range(substitute_n):
            n_indices = [i for i in range(length) if random() < ner_extension.substitute.p]
            if not n_indices: n_indices.append(randint(0, length - 1))
            new_ner, new_fence = substitute_o(ner, fence, n_indices, o_idx)
            values = sample(char_vocab.keys(), len(n_indices))
            new_token = substitute_word(token, n_indices, values)
            if sub_idx_fence:
                sub_idx, sub_fence = sub_idx_fence
                new_sub_idx_fence = substitute_word(sub_idx, n_indices, [char_vocab[i] for i in values], sub_fence)
                yield (len(new_token), new_token, new_ner, new_fence), new_sub_idx_fence
            else:
                yield (len(new_token), new_token, new_ner, new_fence), None

    def __call__(self, idx, token, ner, fence, sub_idx_fence):
        yield from self.insert(idx, token, ner, fence, sub_idx_fence)
        yield from self.delete(idx, token, ner, fence, sub_idx_fence)
        yield from self.substitute(idx, token, ner, fence, sub_idx_fence)

from data.ner_types import break_o
from utils.shell_io import byte_style
class NerDataset(LengthOrderedDataset):
    def __init__(self,
                 fname,
                 ner_extension,
                 char_vocab,
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

        if extra_text_helper:
            _, c2is = v2is['char']
            extra_text_helper = extra_text_helper(text, device, c2is)

        self._aug_size = 0
        if b2is:
            heads = ('bio',)
            columns = dict(token = token, bio = bio)
            factors = None
        else:
            heads = ('ner', 'fence')
            columns = dict(token = token, ner = ner, fence = ner_fence)
            break_o_chunk, break_whole, augment = 0, True, None
            o_idx = n2is('O')
            if ner_extension is not None:
                break_o_chunk = ner_extension.break_o_chunk
                break_whole   = ner_extension.break_whole
                if ner_extension.insert.a + ner_extension.delete.a + ner_extension.substitute.a > 0:
                    assert 'pos' not in v2is
                    print(byte_style('Augmenting with Negative Samples', '2'))
                    augment = NegativeAugment(lengths, ner_extension, char_vocab, o_idx)
                    self._aug_size += len(augment)
            self._break_o = break_o_chunk, break_whole, o_idx
            if break_whole:
                factors = {False: 1 - break_o_chunk, True: break_o_chunk}
            else:
                factors = break_o_chunk > 0
            self._augment_cache = augment, [], []
        if p2is:
            heads = ('token', 'pos') + heads
            columns['pos'] = pos
        else:
            heads = ('token',) + heads
        assert all(len(lengths) == len(col) for col in columns.values())

        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)
        self._columns = columns
        self._device = device

    @property
    def size(self):
        return self._aug_size + super().size

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = {}
        for hd in self.heads:
            if hd == 'length':
                value = length
            else:
                value = self._columns[hd][idx]
                if factor and hd == 'fence':
                    augment, cache, sub_cache = self._augment_cache
                    if augment is not None:
                        for aug_data, aug_sub in augment(idx, sample['token'], sample['ner'], value, helper_outputs):
                            cache.append(aug_data)
                            if aug_sub is not None:
                                sub_cache.append(aug_sub)
                    sample['ner'], value = break_o(sample['ner'], value, *self._break_o)
            sample[hd] = value
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        if 'fence' in self.heads:
            augment, cache, sub_cache = self._augment_cache
            batch += cache
            self._augment_cache = augment, [], []
            if sub_cache and self._extra_text_helper:
                self._extra_text_helper.a_secrete_buffer(sub_cache)
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