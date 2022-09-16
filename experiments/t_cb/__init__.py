from data.penn_types import E_CONTINUE, nccp_data_config
from experiments.t_cb.model import CB, model_type
from experiments.t_cb.operator import CBOperator, train_type

CORPORA = set(E_CONTINUE)

def get_configs(recorder = None):
    if recorder is None:
        return nccp_data_config, model_type, train_type
    
    from data.penn import PennReader
    from data.utils import post_word_base
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    penn = HParams(data_config)
    
    for corp_name, dc in data_config[K_CORP].items():
        readers[corp_name] = PennReader(corp_name,
            HParams(dc, fallback_to_none = True),
            penn.unify_sub,
            penn.nil_pad)
                
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                if (train_ds := reader.loaded_ds.get(mode)) is None:
                    datasets[corp_name] = reader.binary(
                        M_TRAIN, penn.condense_per, penn.batch_size, penn.bucket_len, 0, penn.max_len,
                        penn.sort_by_length, new_factor[corp_name] if new_factor else None)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_binary_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
            else:
                datasets[corp_name] = reader.binary(mode, penn.condense_per, penn.batch_size << 1, 0)
        return datasets
        
    if not penn.condense_per:
        from data.continuous.binary.mp import TriangularDM as dm_cls
    else:
        from data.continuous.binary.mp import TrapezoidalDM as dm_cls

    model, i2vs = post_word_base(CB, model_config, data_config[K_CORP], readers)
    get_dm = lambda num_threads: dm_cls(penn.batch_size << 1, i2vs, num_threads)
    return CBOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)