
from data.stutt import DiscoMultiReader
from data.stutt_types import E_DISCONTINUOUS, xccp_data_config, select_and_split_corpus
from utils.types import M_TRAIN
from utils.param_ops import HParams

from experiments.t_plm_db import get_any_disco
from experiments.t_plm_dm.model import DiscoPlmTree, model_type
from experiments.t_plm_dm.operator import DiscoMultiOperator_lr
from experiments.t_dm.operator import train_type

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    if recorder is None:
        return xccp_data_config, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    corp_name, penn, DatasetHelper, Leaves = get_any_disco(**data_config)
    penn = HParams(penn, fallback_to_none = True)

    model = HParams(model_config)
    data_splits = select_and_split_corpus(corp_name,
                                          penn.source_path,
                                          penn.data_splits.train_set,
                                          penn.data_splits.devel_set,
                                          penn.data_splits.test_set,
                                          False, False) # TODO dep

    reader = DiscoMultiReader(penn.data_path,
                              penn.medium_factor.balanced > 0,
                              penn.unify_sub,
                              penn.continuous_fence_only,
                              data_splits,
                              penn.vocab_size,
                              None,
                              DatasetHelper)
    
    def get_datasets(mode, new_configs = None):
        datasets = {}
        if mode == M_TRAIN:
            train_ds = reader.loaded_ds.get(mode)
            assert new_configs is None, 'should not optuna medium_factor for xbert'
            if train_ds is None:
                datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                                   penn.medium_factor._nested,
                                                   max_len = penn.max_len,
                                                   min_gap = penn.min_gap,
                                                   sort_by_length = penn.sort_by_length,
                                                   inter_2d = train_config.disco_2d_inter_rate > 0 and penn.max_inter_height)
            else:
                from data.backend import post_batch
                datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
        else:
            datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}
    model_config['space_layer']['continuous_fence_only'] = penn.continuous_fence_only
    model = DiscoPlmTree(Leaves, **model_config, **task_params)
    from data.cross.multib import MxDM
    get_dm = lambda num_threads: MxDM(penn.batch_size << 1, reader.i2vs, num_threads)
    return DiscoMultiOperator_lr(model, get_datasets, recorder, reader.i2vs, get_dm, train_config, recorder.evalb)