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
        dc = HParams(dc)
        readers[corp_name] = DTreeReader(corp_name, dc, stutt.unify_sub)
        if dc.disco_2d.inter_rate > 0 and dc.max_interply > 0:
            assert train_config.loss_weight.disco_2d_inter > 0
        if dc.disco_2d.intra_rate > 0:
            assert train_config.loss_weight.disco_2d_intra > 0
    
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                train_ds = reader.loaded_ds.get(mode)
                if train_ds is None:
                    datasets[corp_name] = reader.multib(
                        M_TRAIN,
                        stutt.batch_size,
                        stutt.bucket_len,
                        stutt.continuous_chunk_only,
                        max_len = stutt.max_len,
                        sort_by_length = stutt.sort_by_length,
                        new_factor = new_factor[corp_name] if new_factor else None)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_multib_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, stutt.sort_by_length, stutt.bucket_len, stutt.batch_size)
            else:
                datasets[corp_name] = reader.multib(mode, stutt.batch_size << 1, 0, stutt.continuous_chunk_only)
        return datasets

    from data.utils import post_word_base
    model_config['space_layer']['continuous_chunk_only'] = stutt.continuous_chunk_only
    model, i2vs = post_word_base(DM, model_config, data_config[K_CORP], readers)
    from data.cross.mp import MxDM
    get_dm = lambda num_threads: MxDM(stutt.batch_size << 1, i2vs, num_threads)
    return DMOperater(model, get_datasets, recorder, i2vs, get_dm, train_config, recorder.evalb)
