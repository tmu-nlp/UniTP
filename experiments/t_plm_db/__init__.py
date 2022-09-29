from data.stutt_types import E_DISCONTINUOUS, C_DPTB, C_TIGER
from data.stutt_types import dccp_data_config
from utils.types import M_TRAIN
from utils.param_ops import HParams

from experiments.t_plm_db.model import DiscoPlmTree, model_type
from experiments.t_plm_db.operator import DiscoOperator_lr
from experiments.t_db.operator import train_type

CORPORA = set(E_DISCONTINUOUS)

def get_any_disco(dptb = None, tiger = None):
    from models.plm import XLNetDatasetHelper, XLNetLeaves, GBertDatasetHelper, GBertLeaves
    if dptb is None:
        return C_TIGER, tiger, GBertDatasetHelper, GBertLeaves
    return C_DPTB, dptb, XLNetDatasetHelper, XLNetLeaves

def get_configs(recorder = None):
    if recorder is None:
        return dccp_data_config, model_type, train_type

    data_config, model_config, train_config, _ = recorder.task_specs()
    corp_name, disco, DatasetHelper, Leaves = get_any_disco(**data_config)
    disco = HParams(disco, fallback_to_none = True)
    train_cnf     = disco.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x] if x in E_ORIF5_HEAD else 0): 1}

    reader = DiscoReader(disco.data_path,
                         disco.vocab_size,
                         disco.unify_sub,
                         extra_text_helper = DatasetHelper)
    
    def get_datasets(mode, new_train_cnf = None):
        datasets = {}
        if mode == M_TRAIN:
            train_ds = reader.loaded_ds.get(mode)
            assert new_train_cnf is None, 'should not optuna train_cnf for xbert'
            if train_ds is None:
                datasets[corp_name] = reader.batch(M_TRAIN,
                                                   disco.batch_size,
                                                   disco.bucket_len,
                                                   train_cnf,
                                                   max_len = disco.max_len,
                                                   min_gap = disco.min_gap,
                                                   ply_shuffle = disco.ply_shuffle,
                                                   sort_by_length = disco.sort_by_length)
            else:
                from data.backend import post_batch
                datasets[corp_name] = post_batch(mode, train_ds, disco.sort_by_length, disco.bucket_len, disco.batch_size)
        else:
            datasets[corp_name] = reader.batch(mode, disco.batch_size << 1, 0, non_train_cnf)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}

    model = DiscoPlmTree(Leaves, **model_config, **task_params)
    from data.cross.binary import BxDM
    get_dm = lambda num_threads: BxDM(disco.batch_size << 1, reader.i2vs, num_threads)
    return DiscoOperator_lr(model, get_datasets, recorder, reader.i2vs, get_dm, train_config, recorder.evalb)