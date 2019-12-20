from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, E_ORIF, UNK, NIL, BOS, EOS
from data.io import load_i2vs

SUB = '_SUB'
from data.backend import BaseReader, DataLoader, BatchSpec

class PennReader(BaseReader):
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
            from data.triangle import TriangularDataset
            len_sort_ds = TriangularDataset(self.dir_join, mode, **common_args)
        else:
            from data.trapezoid import TrapezoidDataset
            tree_reader, get_fnames, _, data_splits, trapezoid_height = trapezoid_specs
            len_sort_ds = TrapezoidDataset(trapezoid_height, tree_reader, get_fnames, data_splits[mode], **common_args)

        if mode != M_TRAIN:
            len_sort_ds.plain_mode()
        elif sort_by_length:
            if bucket_length > 0:
                len_sort_ds.increasing_mode(bucket_length)
            else:
                len_sort_ds.plain_mode()
        else:
            len_sort_ds.bucketed_mode(bucket_length)
        di = DataLoader(len_sort_ds, batch_size = batch_size, collate_fn = len_sort_ds.collate_fn, shuffle = mode == M_TRAIN)#, num_workers = 1) # no way to get more!
        return BatchSpec(len(len_sort_ds), di)