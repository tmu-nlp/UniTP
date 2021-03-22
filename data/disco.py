from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, NIL
from data.io import load_i2vs
from data.io import isfile

from data.backend import WordBaseReader, post_batch, defaultdict

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
            labels.append('_SUB')
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
              min_gap        = None,
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
                           min_gap = min_gap,
                           swapper = mode == M_TRAIN and shuffle_swap,
                           extra_text_helper = extra_text_helper,
                           train_indexing_cnn = train_indexing_cnn)

        from data.cross.dataset import StaticCrossDataset
        len_sort_ds = StaticCrossDataset(self.dir_join, mode, **common_args)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)


class DiscoMultiReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 has_greedy_sub,
                 unify_sub,
                 data_splits,
                 vocab_size = None,
                 word_trace = False,
                 extra_text_helper = None):
        i2vs = load_i2vs(vocab_dir, 'word tag label'.split())
        oovs = {}
        labels = i2vs['label']
        if has_greedy_sub:
            from utils.shell_io import byte_style
            print(byte_style('+ greedy_subs', '2'))
        if unify_sub:
            labels = [t for t in labels if t[0] not in '#_']
            oovs['label'] = len(labels)
            labels.append('_SUB' if has_greedy_sub else '#SUB')
            i2vs['label'] = labels
        elif not has_greedy_sub: # MAry does not have binarization
            i2vs['label'] = [t for t in labels if t[0] != '_']
            
        super(DiscoMultiReader, self).__init__(vocab_dir, vocab_size, True, i2vs, oovs)

        v2is = self.v2is
        v2is = v2is['token'][1], v2is['tag'][1], v2is['label'][1]
        samples = defaultdict(list)
        corpus, in_train_set, in_devel_set, in_test_set, from_tree_fn = data_splits
        from tqdm import tqdm
        for fid, tree, dep in tqdm(corpus, 'Loading corpus'):
            try:
                keeper = from_tree_fn(tree, v2is, dep, has_greedy_sub)
            except AssertionError: # 2 multi-attatchment in tiger test set
                continue
            if in_train_set(fid):
                samples[M_TRAIN].append(keeper)
            elif in_devel_set(fid):
                samples[M_DEVEL].append(keeper)
            elif in_test_set(fid):
                samples[M_TEST].append(keeper)
        self._load_options = samples, word_trace, extra_text_helper

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              medium_factors = None,
              min_len        = 1,
              max_len        = None,
              min_gap        = 0,
              sort_by_length = True):
        from data.cross.dataset import DynamicCrossDataset
        samples, word_trace, extra_text_helper = self._load_options
        len_sort_ds = DynamicCrossDataset(samples[mode],
                                          self.device,
                                          medium_factors,
                                          min_len,
                                          max_len,
                                          min_gap,
                                          extra_text_helper)
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)