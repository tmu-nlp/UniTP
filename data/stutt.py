from utils.types import M_TRAIN, M_DEVEL, M_TEST, beta_type
from data.io import load_i2vs
from data import USUB
from data.utils import ParsingReader
from data.dataset import post_batch
from data.penn_types import E_CONTINUE
from data.cross import Signal
from data.cross.dataset import DiscontinuousDataset
Signal.set_binary()
Signal.set_multib()

class DTreeReader(ParsingReader):
    def __init__(self, corp_name, stutt,
                 unify_sub   = True,
                 nil_as_pads = True,
                 extra_text_helper = None):

        if corp_name[1:] in E_CONTINUE:
            from data.penn_types import select_and_split_corpus
            from_fn = Signal.from_disco_penn
            reader, corp_split, _ = select_and_split_corpus(
                corp_name[1:],
                stutt.source_path,
                stutt.build_params.train_set,
                stutt.build_params.devel_set,
                stutt.build_params.test_set)
        else:
            from data.stutt_types import select_and_split_corpus
            from_fn = Signal.from_tiger_graph
            reader = None
            corp_split = select_and_split_corpus(
                corp_name,
                stutt.source_path,
                stutt.build_params.train_set,
                stutt.build_params.devel_set,
                stutt.build_params.test_set)
        
        self._load_options = reader, corp_split, from_fn, stutt, extra_text_helper
        super().__init__(stutt, unify_sub, nil_as_pads)

    def binary(self, mode, batch_size, bucket_length,
               min_len        = 0,
               max_len        = None,
               sort_by_length = True,
               new_factor     = None):
        reader, corp_split, from_fn, stutt, extra_text_helper = self._load_options
        esub = msub = min_gap = 0
        binarization = None
        if mode == M_TRAIN:
            min_gap = stutt.min_gap
            if new_factor is None:
                binarization = beta_type(stutt.binarization)
                esub, msub = stutt.esub, stutt.msub
            else:
                binarization, esub, msub = new_factor
        b_pad_shuffle = self.paddings, stutt.ply_shuffle

        self.loaded_ds[mode] = ds = DiscontinuousDataset(True, reader, corp_split[mode], from_fn, self.v2is,
            binarization, esub, msub, b_pad_shuffle, min_gap, min_len, max_len, extra_text_helper)
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)

    def multib(self, mode, batch_size, bucket_length,
               continuous_chunk_only,
               min_len        = 0,
               max_len        = None,
               sort_by_length = True,
               new_factor     = None):
        reader, corp_split, from_fn, stutt, extra_text_helper = self._load_options
        esub = msub = min_gap = 0
        medoid = None
        intra_rate, inter_rate = stutt.disco_2d.intra_rate, stutt.disco_2d.inter_rate
        if mode == M_TRAIN:
            min_gap = stutt.min_gap
            if new_factor is None:
                medoid = stutt.medoid._nested
                esub, msub = stutt.esub, stutt.msub
            elif len(new_factor) == 3:
                medoid, esub, msub = new_factor
            else:
                medoid, esub, msub, intra_rate, inter_rate = new_factor
        m_fence_2d = continuous_chunk_only, stutt.max_interply, intra_rate, inter_rate

        self.loaded_ds[mode] = ds = DiscontinuousDataset(False, reader, corp_split[mode], from_fn, self.v2is,
            medoid, esub, msub, m_fence_2d, min_gap, min_len, max_len, extra_text_helper)
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)