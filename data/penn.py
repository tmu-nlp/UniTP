from utils.types import M_TRAIN, beta_type
from data.io import load_i2vs
from data.dataset import post_batch
from data.utils import ParsingReader
from data import NIL
from data.penn_types import select_and_split_corpus
from data.continuous.dataset import ContinuousDataset

class PennReader(ParsingReader):
    def __init__(self, corp_name, penn,
                 unify_sub   = True,
                 nil_as_pads = True,
                 extra_text_helper = None):

        reader, fileid_split, from_fn = select_and_split_corpus(
            corp_name,
            penn.source_path,
            penn.build_params.train_set,
            penn.build_params.devel_set,
            penn.build_params.test_set)
        self._load_options = reader, fileid_split, from_fn, penn, extra_text_helper

        super().__init__(penn, unify_sub, nil_as_pads)

    def binary(self, mode, condense_per, batch_size, bucket_length,
               min_len        = 0,
               max_len        = None,
               sort_by_length = True,
               new_factor     = None):
        reader, fileid_split, from_fn, penn, extra_text_helper = self._load_options
        esub = msub = 0
        binarization = None
        if mode == M_TRAIN:
            if new_factor is None:
                binarization = beta_type(penn.binarization)
                esub, msub = penn.esub, penn.msub
            else:
                binarization, esub, msub = new_factor

        self.loaded_ds[mode] = ds = ContinuousDataset(True, reader, fileid_split[mode], from_fn,
            self.v2is, False, self.paddings, binarization, esub, msub,
            condense_per, min_len, max_len, extra_text_helper)        
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)

    def multib(self, mode, batch_size, bucket_length,
               min_len        = 0,
               max_len        = None,
               sort_by_length = True,
               new_factor     = None):
        reader, fileid_split, from_fn, penn, extra_text_helper = self._load_options
        esub = msub = 0
        factor = None
        if mode == M_TRAIN:
            factor = 'dummy:)'
            if new_factor is None:
                esub, msub = penn.esub, penn.msub
            else:
                esub, msub = new_factor
        self.loaded_ds[mode] = ds = ContinuousDataset(False, reader, fileid_split[mode], from_fn,
            self.v2is, penn.token == 'char', self.paddings, factor, esub, msub,
            True, min_len, max_len, extra_text_helper)     
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)


from utils.types import false_type
from utils.types import train_batch_size, train_max_len, train_bucket_len
tokenization_config = dict(lower_case       = false_type,
                           batch_size       = train_batch_size,
                           max_len          = train_max_len,
                           bucket_len       = train_bucket_len,
                           sort_by_length   = false_type)

from collections import Counter
from data.noise import SequenceBaseReader
# from utils.param_ops import dict_print
class LexiconReader(SequenceBaseReader):
    def __init__(self,
                 vocab_dir,
                 lower_case = False):
        i2vs = load_i2vs(vocab_dir, ('word',))
        word = i2vs.pop('word')
        assert word.pop(0) == NIL
        char = Counter()
        data = []
        for w in word:
            if lower_case:
                w = w.lower()
            char += Counter(w)
            data.append(w)
        i2vs['token'] = [NIL] + sorted(char.keys())
        # print(dict_print({k:char[k] for k in sorted(char, key = char.get, reverse = True)}))
        super(LexiconReader, self).__init__(i2vs)
        self._char_data = char, data

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              noise_specs,
              factors,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        from data.noise import CharDataset

        if noise_specs is None:
            assert sum(factors[k] for k in 'swap insert replace delete'.split() if k in factors) == 0, 'Need specs!'
        char, data = self._char_data
        len_sort_ds = CharDataset(char, data, self.v2is, noise_specs, factors, min_len, max_len)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)