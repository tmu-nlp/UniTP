from utils.types import fill_placeholder, M_TRAIN, M_DEVEL, M_TEST, E_ORIF, UNK, NIL, BOS, EOS
from data.io import load_i2vs

SUB = '_SUB'
from data.backend import BaseReader, TriangularDataset, DataLoader, BatchSpec

class PennReader(BaseReader):
    def __init__(self,
                 vocab_dir,
                 vocab_size,
                 load_label,
                 unify_sub   = True,
                 load_ftags  = False,
                 nil_as_pads = True):
        self._load_options = load_label, load_ftags
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
        load_label, load_ftags = self._load_options

        assert mode in (M_TRAIN, M_DEVEL, M_TEST)
        if load_label:
            assert isinstance(binarization, dict)
            binarization = {k:v for k,v in binarization.items() if v}
            assert abs(sum(v for v in binarization.values()) - 1) < 1e-10
        else:
            assert binarization is None

        ds = TriangularDataset(self.dir_join, mode, self.v2is, self.paddings, self.device, binarization, min_len = min_len, max_len = max_len)
        if mode != M_TRAIN:
            ds.plain_mode()
        elif sort_by_length:
            if bucket_length > 0:
                ds.increasing_mode(bucket_length)
            else:
                ds.plain_mode()
        else:
            ds.bucketed_mode(bucket_length)
        di = DataLoader(ds, batch_size = batch_size, collate_fn = ds.collate_fn, shuffle = mode == M_TRAIN)#, num_workers = 1) # no way to get more!
        return BatchSpec(len(ds), di)