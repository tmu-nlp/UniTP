from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, NIL
from data.io import load_i2vs
from data.io import isfile
from torch import load, save

SUB = '_SUB'
from data.backend import WordBaseReader, post_batch

class DiscoReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size  = None,
                 unify_sub   = True,
                 extra_text_helper = None):
        self._load_options = True, extra_text_helper, False
        vocabs = 'word tag label'
        i2vs = load_i2vs(vocab_dir, vocabs.split())
        oovs = {}
        if unify_sub:
            labels = [t for t in i2vs['label'] if t[0] not in '#_']
            oovs['label'] = len(labels)
            labels.append(SUB)
            i2vs['label'] = labels
        super(DiscoReader, self).__init__(vocab_dir, vocab_size, True, i2vs, oovs)

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              binarization   = None,
              shuffle_swap   = False,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        load_label, extra_text_helper, train_indexing_cnn = self._load_options
        assert mode in (M_TRAIN, M_DEVEL, M_TEST)

        if load_label:
            assert isinstance(binarization, dict)
            binarization = {k:v for k,v in binarization.items() if v}
            assert abs(sum(v for v in binarization.values()) - 1) < 1e-10
        else:
            assert binarization is None

        common_args = dict(field_v2is = self.v2is,
                           device = self.device,
                           factors = binarization,
                           min_len = min_len,
                           max_len = max_len, 
                           swapper = mode == M_TRAIN and shuffle_swap,
                           extra_text_helper = extra_text_helper,
                           train_indexing_cnn = train_indexing_cnn)

        from data.cross.dataset import CrossDataset
        len_sort_ds = CrossDataset(self.dir_join, mode, **common_args)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)