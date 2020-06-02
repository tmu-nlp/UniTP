from data.io import load_i2vs
from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, NIL
from data.backend import WordBaseReader, post_batch
from data.stan_types import split_files
split_files = {v:k for k,v in split_files.items()}

class StanReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size,
                 nil_as_pads = True,
                 nil_is_neutral = True,
                 trapezoid_specs   = None,
                 extra_text_helper = None,
                 extra_vocab = None):
        i2vs = load_i2vs(vocab_dir, 'word polar'.split())
        oovs = {}
        if nil_is_neutral:
            polar = i2vs['polar']
            polar.pop(polar.index(NIL))
            polar.pop(polar.index('2'))
            polar.insert(0, '2')
            oovs['polar'] = 0
        else:
            assert i2vs['polar'][0] == NIL
        super(StanReader, self).__init__(vocab_dir, vocab_size, nil_as_pads, i2vs, oovs, extra_vocab)
        self._load_options = trapezoid_specs, extra_text_helper

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        assert mode in (M_TRAIN, M_DEVEL, M_TEST)
        trapezoid_specs, extra_text_helper = self._load_options

        common_args = dict(field_v2is = self.v2is,
                           paddings = self.paddings,
                           device = self.device,
                           factors = None,
                           min_len = min_len,
                           max_len = max_len, 
                           extra_text_helper = extra_text_helper)

        if trapezoid_specs is None:
            from data.triangle.dataset import TriangularDataset
            len_sort_ds = TriangularDataset(self.dir_join, mode, **common_args)
        else:
            from data.trapezoid.dataset import TrapezoidDataset
            from os.path import join
            trapezoid_height, sstb_path = trapezoid_specs
            sstb_path = join(sstb_path, f'{split_files[mode]}.txt')
            len_sort_ds = TrapezoidDataset.from_stan(sstb_path, trapezoid_height, **common_args)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)