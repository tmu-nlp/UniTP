from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, E_ORIF, UNK, NIL, BOS, EOS
from utils.param_ops import change_key
from data.io import load_i2vs

SUB = '_SUB'
from data.backend import WordBaseReader, DataLoader, post_batch

class PennReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size,
                 load_label,
                 unify_sub   = True,
                 load_ftags  = False,
                 nil_as_pads = True,
                 trapezoid_specs   = None,
                 extra_text_helper = None):
        self._load_options = load_label, load_ftags, trapezoid_specs, extra_text_helper
        vocabs = 'word tag'
        if load_label:
            vocabs += ' label'
            if load_ftags:
                vocabs += ' ftag'
        i2vs = load_i2vs(vocab_dir, vocabs.split())
        change_key(i2vs, 'word', 'token')
        oovs = {}
        if load_label and unify_sub:
            labels = [t for t in i2vs['label'] if t[0] not in '#_']
            oovs['label'] = len(labels)
            labels.append(SUB)
            i2vs['label'] = labels
        super(PennReader, self).__init__(vocab_dir, vocab_size, nil_as_pads, i2vs, oovs)

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              binarization   = None,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        load_label, load_ftags, trapezoid_specs, extra_text_helper = self._load_options

        assert mode in (M_TRAIN, M_DEVEL, M_TEST)

        if load_label:
            assert isinstance(binarization, dict)
            binarization = {k:v for k,v in binarization.items() if v}
            assert abs(sum(v for v in binarization.values()) - 1) < 1e-10
        else:
            assert binarization is None

        common_args = dict(field_v2is = self.v2is,
                           paddings = self.paddings,
                           device = self.device,
                           factors = binarization,
                           min_len = min_len,
                           max_len = max_len, 
                           extra_text_helper = extra_text_helper)

        if not load_label or trapezoid_specs is None:
            from data.triangle.dataset import TriangularDataset
            len_sort_ds = TriangularDataset(self.dir_join, mode, **common_args)
        else:
            from data.trapezoid.dataset import TrapezoidDataset
            tree_reader, get_fnames, _, data_splits, trapezoid_height = trapezoid_specs
            len_sort_ds = TrapezoidDataset(trapezoid_height, tree_reader, get_fnames, data_splits[mode], **common_args)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)

from utils.types import false_type, true_type
from utils.types import train_batch_size, train_max_len, train_bucket_len
tokenization_config = dict(lower_case       = false_type,
                           batch_size       = train_batch_size,
                           max_len          = train_max_len,
                           bucket_len       = train_bucket_len,
                           sort_by_length   = false_type)

from collections import Counter
from data.backend import SequenceBaseReader
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
        super(LexiconReader, self).__init__(vocab_dir, i2vs)
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
        len_sort_ds = CharDataset(char, data, self.v2is, noise_specs, factors, self.device, min_len, max_len)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)