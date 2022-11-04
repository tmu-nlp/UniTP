from data.vocab import VocabKeeper
from data.dataset import LengthOrderedDataset, post_batch
from data.dataset import pad_tag_like_nil, erect_split_more, fill_bool_tensor
from data.ner_types import read_dataset, remove_bio_prefix, bio_to_tree
from data.io import load_i2vs, join, get_fasttext
from utils.types import M_TRAIN
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
                 extra_text_helper = None):
        if with_bi_prefix:
            vocabs = ['word', 'pos', 'bio']
        else:
            vocabs = ['word', 'pos', 'ner']
        i2vs = load_i2vs(ner.local_path, *vocabs)

        change_key(i2vs, 'word', 'token')
        change_key(i2vs, 'pos', 'tag')
        if with_bi_prefix:
            change_key(i2vs, 'bio', 'label')
        else:
            change_key(i2vs, 'ner', 'label')

        def token_fn(): # token_type
            weights = get_fasttext(join(ner.local_path, 'word.vec'))
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
        self._load_options = ner, with_bi_prefix, extra_text_helper, char_vocab
        super().__init__(('token', 'tag', 'label'), i2vs, {}, {}, weight_fn = token_fn)

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              min_len        = 0,
              max_len        = None,
              sort_by_length = True):
        ner, with_bi_prefix, extra_text_helper, char_vocab = self._load_options

        len_sort_ds = NerDataset(join(ner.source_path, ner.build_params._nested[mode + '_set']),
                                 mode == M_TRAIN,
                                 with_bi_prefix,
                                 ner.extension._nested,
                                 char_vocab,
                                 self.v2is,
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
                 train_mode,
                 with_bi_prefix,
                 ner_extension,
                 char_vocab,
                 v2is,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        lengths, token, tag, text, = [], [], [], []
        label, ner_fence, tree = [], [], []
        if (has_syntax := ('with_o_chunk' in ner_extension)):
            n_files = 4
        else:
            n_files = 3
        for largs in read_dataset(fname, n_files):
            if has_syntax:
                wd, pt, chk, bi = largs
                chk = chk if ner_extension['with_o_chunk'] else None
                fc, nt = remove_bio_prefix(bi, chk)
            else:
                wd, pt, bi = largs
                fc, nt = remove_bio_prefix(bi)
            lb = bi if with_bi_prefix else nt
            lengths.append(len(wd))
            text.append(wd)
            ner_fence.append(fc)
            label.append([v2is.label(x) for x in lb])
            token.append([v2is.token(x) for x in wd])
            tag.append([v2is.tag(x) for x in pt])
            tree.append(bio_to_tree(wd, bi, pt))

        if extra_text_helper:
            _, c2is = v2is['char']
            extra_text_helper = extra_text_helper(text, self.device, c2is)

        factors = None
        self._aug_size = 0
        if train_mode:
            if with_bi_prefix:
                heads = 'tree', 'token', 'tag', 'label'
            else:
                heads = 'tree', 'token', 'tag', 'label', 'fence'
            if not has_syntax:
                o_idx = v2is.label('O')
                break_o_chunk, break_whole, augment = 0, True, None
                if ner_extension is not None:
                    break_o_chunk = ner_extension.break_o_chunk
                    break_whole   = ner_extension.break_whole
                    if ner_extension.insert.a + ner_extension.delete.a + ner_extension.substitute.a > 0:
                        print(byte_style('Augmenting with Negative Samples', '2'))
                        augment = NegativeAugment(lengths, ner_extension, char_vocab, o_idx)
                        self._aug_size += len(augment)
                self._break_o = break_o_chunk, break_whole, o_idx
                if break_whole:
                    factors = {False: 1 - break_o_chunk, True: break_o_chunk}
                else:
                    factors = break_o_chunk > 0
                self._augment_cache = augment, [], []
        else:
            heads = 'tree', 'token'

        field_columns = dict(n_layers = int(not with_bi_prefix),
                             stop_at_nth_layer = True)
        super().__init__(heads, None, lengths, factors, min_len, max_len, extra_text_helper)
        self._args = train_mode, has_syntax, field_columns, tree, token, tag, label, ner_fence

    @property
    def size(self):
        return self._aug_size + super().size

    def at_idx(self, idx, factor, helper_outputs):
        train_mode, has_syntax, _, tree, token, tag, label, fence = self._args
        token  = token[idx]
        sample = [tree[idx], token]
        if train_mode:
            label = label[idx]
            if 'fence' in self.heads:
                fence = fence[idx]
                if not has_syntax and factor:
                    augment, cache, sub_cache = self._augment_cache
                    if augment is not None:
                        for aug_data, aug_sub in augment(idx, token, label, fence, helper_outputs):
                            cache.append(aug_data)
                            if aug_sub is not None:
                                sub_cache.append(aug_sub)
                    label, fence = break_o(label, fence, *self._break_o)
                sample += [tag[idx], label, [fence]]
            else:
                sample += [tag[idx], label]
        return tuple(sample)

    def _collate_fn(self, batch, length, segment):
        field_columns = {}
        train_mode, has_syntax, fc = self._args[:3]
        indice_args = dict(device = self.device, dtype = torch.long)
        field_columns = dict(length = length, **fc)
        max_token_len = length.max()
        without_bio = 'fence' in self.heads

        if not has_syntax and without_bio:
            augment, cache, sub_cache = self._augment_cache
            batch += cache
            self._augment_cache = augment, [], []
            if sub_cache and self._extra_text_helper:
                self._extra_text_helper.a_secrete_buffer(sub_cache)

        field_columns['tree'] = batch.tree
        field_columns['token'] = torch.as_tensor(pad_tag_like_nil(batch.token, max_token_len), **indice_args)
        if train_mode:
            field_columns['tag'] = torch.as_tensor(pad_tag_like_nil(batch.tag, max_token_len), **indice_args)
            if without_bio:
                max_label_len = max(len(x) for x in batch.label)
                field_columns['label'] = torch.as_tensor(pad_tag_like_nil(batch.label, max_label_len), **indice_args)

                max_fence_len = max_token_len + 1
                field_columns['chunk'] = sig = torch.zeros(len(length), max_fence_len, dtype = torch.bool, device = self.device)
                fill_bool_tensor(erect_split_more(batch.fence, [max_fence_len], 0), sig, True, indice_args)
                field_columns['batch_segment'] = [max_token_len, max_label_len]
            else:
                field_columns['batch_segment'] = [max_token_len]
                field_columns['label'] = torch.as_tensor(pad_tag_like_nil(batch.label, max_token_len), **indice_args)
                
        return field_columns