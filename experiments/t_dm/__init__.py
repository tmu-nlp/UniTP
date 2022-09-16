from data.stutt import DiscoMultiReader
from data.stutt_types import xccp_data_config, E_DISCONTINUOUS
from data.stutt_types import select_and_split_corpus
from utils.types import M_TRAIN
from utils.file_io import parpath
from utils.param_ops import HParams

from experiments.t_dm.model import DiscoMultiRnnTree, model_type
from experiments.t_dm.operator import DiscoMultiOperator, train_type

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    if recorder is None:
        return xccp_data_config, model_type, train_type
    
    from data.utils import pre_word_base, post_word_base
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    chelper = pre_word_base(model_config)
    for corp_name in data_config:
        penn = HParams(data_config[corp_name], fallback_to_none = True)
        data_splits = select_and_split_corpus(corp_name,
                                              penn.source_path,
                                              penn.data_splits.train_set,
                                              penn.data_splits.devel_set,
                                              penn.data_splits.test_set,
                                              binary = False,
                                              read_dep = parpath(penn.data_path) if penn.medium_factor.others.head else None)

        readers[corp_name] = DiscoMultiReader(
            penn.data_path,
            penn.medium_factor.balanced > 0,
            penn.unify_sub,
            penn.continuous_fence_only,
            data_splits,
            penn.vocab_size,
            None,
            chelper)
    
    def get_datasets(mode, new_configs = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                train_ds = reader.loaded_ds.get(mode)
                if train_ds is None:
                    if train_config.disco_2d_inter_rate > 0:
                        assert train_config.loss_weight.disco_2d_inter > 0
                    if train_config.disco_2d_intra_rate > 0:
                        assert train_config.loss_weight.disco_2d_intra > 0
                    if isinstance(new_configs, tuple) and len(new_configs) == 2:
                        new_medium_factor, max_inter_height = new_configs
                    else:
                        new_medium_factor = new_configs if new_configs else penn.medium_factor._nested
                        max_inter_height = penn.max_inter_height
                    datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                                    new_medium_factor,
                                                    max_len = penn.max_len,
                                                    min_gap = penn.min_gap,
                                                    sort_by_length = penn.sort_by_length,
                                                    inter_2d = train_config.disco_2d_inter_rate > 0 and max_inter_height)
                else:
                    from data.backend import post_batch
                    assert isinstance(new_configs, tuple) and len(new_configs) == 2
                    train_ds.reset_factors(*new_configs)
                    datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
            else:
                datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    model_config['space_layer']['continuous_fence_only'] = penn.continuous_fence_only
    model, i2vs = post_word_base(DiscoMultiRnnTree, model_config, data_config, readers)
    from data.cross.multib import MxDM
    get_dm = lambda num_threads: MxDM(penn.batch_size << 1, i2vs, num_threads)
    return DiscoMultiOperator(model, get_datasets, recorder, i2vs, get_dm, train_config, recorder.evalb)
