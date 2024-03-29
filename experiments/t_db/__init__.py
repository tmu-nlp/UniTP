from data.stutt_types import E_DISCONTINUOUS, dccp_data_config, C_TIGER, C_DPTB

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    from experiments.t_db.model import model_type
    from experiments.t_db.operator import train_type
    if recorder is None:
        return dccp_data_config, model_type, train_type
    return make_instance(recorder)

def make_instance(recorder, use_plm = False):
    from data.stutt import DTreeReader
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams
    from data.utils import post_word_base, parsing_pnames

    data_config, model_config, train_config, _ = recorder.task_specs()
    stutt = HParams(data_config)
    readers = {}

    if use_plm:
        if len(data_config[K_CORP]) == 1:
            if C_TIGER in data_config[K_CORP]:
                from models.plm import GBertDatasetHelper as DatasetHelper
                from experiments.t_plm_db.model import GBertDB as DB
            elif C_DPTB in data_config[K_CORP]:
                from models.plm import XLNetDatasetHelper as DatasetHelper
                from experiments.t_plm_db.model import XLNetDB as DB
            else:
                raise ValueError(', '.join(data_config[K_CORP].keys()))
        else:
            raise ValueError(', '.join(data_config[K_CORP].keys()))
        from experiments.t_plm_db.operator import DBOperator_lr as DBOperator
        pnames = parsing_pnames[2:]
    else:
        from experiments.t_db.operator import DBOperator
        from experiments.t_db.model import DB
        DatasetHelper = None
        pnames = parsing_pnames
    
    for corp_name, dc in data_config[K_CORP].items():
        dc['token'] = 'word'
        readers[corp_name] = DTreeReader(corp_name,
            HParams(dc),
            stutt.unify_sub,
            stutt.nil_pad,
            DatasetHelper)

    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                train_ds = reader.loaded_ds.get(mode)
                if train_ds is None:
                    datasets[corp_name] = reader.binary(
                        M_TRAIN,
                        stutt.batch_size,
                        stutt.bucket_len,
                        max_len = stutt.max_len,
                        sort_by_length = stutt.sort_by_length,
                        new_factor = new_factor[corp_name] if new_factor else None)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_binary_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, stutt.sort_by_length, stutt.bucket_len, stutt.batch_size)
            else:
                datasets[corp_name] = reader.binary(mode, stutt.batch_size << 1, 0)
        return datasets

    model, i2vs = post_word_base(DB, model_config, data_config[K_CORP], readers, pnames)
    from data.cross.mp import BxDM
    get_dm = lambda num_threads: BxDM(stutt.batch_size << 1, i2vs, num_threads)
    return DBOperator(model, get_datasets, recorder, i2vs, get_dm, train_config, recorder.evalb)