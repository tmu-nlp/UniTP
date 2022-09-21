from data.penn_types import E_CONTINUE, accp_data_config
from experiments.t_cm.model import CM, model_type
from experiments.t_cm.operator import CMOperator, train_type

CORPORA = set(E_CONTINUE)

def get_configs(recorder = None):
    if recorder is None:
        return accp_data_config, model_type, train_type
    
    from data.penn import PennReader
    from data.utils import post_word_base
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams

    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    penn = HParams(data_config)
    any_char_as_token = False
    
    for corp_name, dc in data_config[K_CORP].items():
        corp_spec = HParams(dc)
        readers[corp_name] = PennReader(corp_name, corp_spec, penn.unify_sub)
        if corp_spec.token == 'char':
            any_char_as_token = True

    if any_char_as_token:
        from data.continuous import Signal
        Signal.set_char()
    model_config['chunk_layer']['char_chunk'] = any_char_as_token
    
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                if (train_ds := reader.loaded_ds.get(mode)) is None:
                    datasets[corp_name] = reader.multib(
                        M_TRAIN,
                        penn.batch_size,
                        penn.bucket_len,
                        max_len = penn.max_len,
                        sort_by_length = penn.sort_by_length, 
                        new_factor = new_factor[corp_name] if new_factor else None)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_multib_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
            else:
                datasets[corp_name] = reader.multib(mode, penn.batch_size << 1, 0)
        return datasets

    model, i2vs = post_word_base(CM, model_config, data_config[K_CORP], readers)
    from data.continuous.multib.mp import MultibDM
    get_dm = lambda num_threads: MultibDM(penn.batch_size << 1, i2vs, num_threads)
    return CMOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)