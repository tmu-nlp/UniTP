from utils.types import M_TRAIN, M_DEVEL, M_TEST, F_RAND_CON, beta_tuple
from data.io import load_i2vs

from data.backend import WordBaseReader, post_batch, defaultdict, CharTextHelper, add_char_from_word

def beta(k, v):
    if k == F_RAND_CON:
        return v if isinstance(v, (tuple, bool)) else beta_tuple(v)
    return v

class DiscoReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size  = None,
                 unify_sub   = True,
                 extra_text_helper = None):
        self._load_options = True, extra_text_helper, False
        vocabs = 'word tag label'
        i2vs = load_i2vs(vocab_dir, vocabs.split())
        if extra_text_helper is CharTextHelper:
            add_char_from_word(i2vs)
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
              ply_shuffle    = None,
              min_len        = 2, # to prevent some errors
              max_len        = None,
              min_gap        = None,
              sort_by_length = True):
        load_label, extra_text_helper, train_indexing_cnn = self._load_options
        assert mode in (M_TRAIN, M_DEVEL, M_TEST)

        if load_label:
            assert isinstance(binarization, dict)
            if binarization.get(F_RAND_CON):
                binarization = {k:beta(k,v) for k,v in binarization.items() if k.startswith(F_RAND_CON)}
            else:
                binarization = {k:v for k,v in binarization.items() if not k.startswith(F_RAND_CON) and v > 0}
                assert abs(sum(binarization.values()) - 1) < 1e-10
        else:
            assert binarization is None

        common_args = dict(field_v2is = self.v2is,
                           factors = binarization,
                           min_len = min_len,
                           max_len = max_len,
                           min_gap = min_gap,
                           ply_shuffle = mode == M_TRAIN and ply_shuffle,
                           extra_text_helper = extra_text_helper,
                           train_indexing_cnn = train_indexing_cnn)

        from data.cross.dataset import BinaryDataset
        self.loaded_ds[mode] = ds = BinaryDataset(self.dir_join, mode, **common_args)
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)


from sys import stderr
from utils.file_io import basename
class DiscoMultiReader(WordBaseReader):
    def __init__(self,
                 vocab_dir,
                 has_greedy_sub,
                 unify_sub,
                 continuous_fence_only,
                 data_splits,
                 vocab_size = None,
                 word_trace = False,
                 extra_text_helper = None):
        i2vs = load_i2vs(vocab_dir, 'word tag label'.split())
        if extra_text_helper is CharTextHelper:
            add_char_from_word(i2vs)
        oovs = {}
        labels = i2vs['label']
        if has_greedy_sub:
            from utils.shell_io import byte_style
            print(byte_style(basename(vocab_dir).upper() + ' + balancing subs', '2'), file = stderr)
        if unify_sub:
            labels = [t for t in labels if t[0] not in '#_']
            oovs['label'] = len(labels)
            labels.append('_SUB' if has_greedy_sub else '#SUB')
            i2vs['label'] = labels
        elif not has_greedy_sub: # MAry does not have binarization
            i2vs['label'] = [t for t in labels if t[0] != '_']
            
        super(DiscoMultiReader, self).__init__(vocab_dir, vocab_size, True, i2vs, oovs)

        v2is = self.v2is
        c2i = v2is['char'][1] if 'char' in v2is else None
        v2is = v2is['token'][1], v2is['tag'][1], v2is['label'][1]
        samples = defaultdict(list)
        corpus, in_train_set, in_devel_set, in_test_set, from_tree_fns = data_splits
        for fid, tree, dep in corpus:
            if in_train_set(fid):
                samples[M_TRAIN].append((tree, dep))
            elif in_devel_set(fid):
                samples[M_DEVEL].append((tree, dep))
            elif in_test_set(fid):
                samples[M_TEST].append((tree, dep))
        self._load_options = from_tree_fns, v2is, has_greedy_sub, continuous_fence_only, word_trace, samples, extra_text_helper, c2i

    def batch(self,
              mode,
              batch_size,
              bucket_length,
              medium_factors = None,
              sort_by_length = True,
              **kwargs):
        from data.cross.dataset import MultibDataset
        from tqdm import tqdm
        (from_tree_fn, v2is, has_greedy_sub, continuous_fence_only, word_trace,
         samples, extra_text_helper, c2i) = self._load_options
        signals = []
        errors = defaultdict(int)
        for tree, dep in tqdm(samples[mode], f'Load {mode.title()}set'):
            try:
                signals.append(from_tree_fn(tree, v2is, dep)) # has_greedy_sub
            except AssertionError as ae:
                errors[ae.args[0]] += 1
        if errors:
            print(errors)
        len_sort_ds = MultibDataset(signals,
                                    medium_factors,
                                    extra_text_helper,
                                    c2i,
                                    continuous_fence_only,
                                    **kwargs)
        self.loaded_ds[mode] = len_sort_ds
        return post_batch(mode, len_sort_ds, sort_by_length, bucket_length, batch_size)