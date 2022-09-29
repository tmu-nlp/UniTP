from data.stutt_types import xccp_data_config, E_DISCONTINUOUS

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    from experiments.t_dm.model import DM, model_type
    from experiments.t_dm.operator import DMOperater, train_type
    if recorder is None:
        return xccp_data_config, model_type, train_type
    
    from data.stutt import DTreeReader
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams

    data_config, model_config, train_config, _ = recorder.task_specs()
    stutt = HParams(data_config)
    readers = {}
    
    for corp_name, dc in data_config[K_CORP].items():
        dc['token'] = 'word'
        readers[corp_name] = DTreeReader(corp_name,
            HParams(dc),
            stutt.unify_sub,
            stutt.nil_pad)
            
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