from data.stutt import DiscoReader
from data.stutt_types import E_DISCONTINUOUS, dccp_data_config
from utils.types import M_TRAIN
from utils.param_ops import HParams

from experiments.t_db.model import DiscoRnnTree, model_type
from experiments.t_db.operator import DiscoOperator, train_type

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    if recorder is None:
        return dccp_data_config, model_type, train_type
    
    from data.utils import pre_word_base, post_word_base
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    chelper = pre_word_base(model_config)
    train_cnf, non_train_cnf = {}, {}
    for corp_name in data_config:
        disco = HParams(data_config[corp_name], fallback_to_none = True)
        readers[corp_name] = DiscoReader(
            disco.data_path,
            disco.vocab_size,
            disco.unify_sub,
            chelper)
        train_cnf[corp_name] = cnf = disco.binarization._nested
        non_train_cnf[corp_name] = {max(cnf, key = lambda x: cnf[x] if x in E_ORIF5_HEAD else 0): 1}
    
    def get_datasets(mode, new_train_cnf = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                train_ds = reader.loaded_ds.get(mode)
                if train_ds is None:
                    datasets[corp_name] = reader.batch(
                        M_TRAIN,
                        disco.batch_size,
                        disco.bucket_len,
                        new_train_cnf or train_cnf[corp_name],
                        max_len = disco.max_len,
                        min_gap = disco.min_gap,
                        ply_shuffle = disco.ply_shuffle,
                        sort_by_length = disco.sort_by_length)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_factors(new_train_cnf)
                    datasets[corp_name] = post_batch(mode, train_ds, disco.sort_by_length, disco.bucket_len, disco.batch_size)
            else:
                datasets[corp_name] = reader.batch(mode, disco.batch_size << 1, 0, non_train_cnf[corp_name])
        return datasets

    model, i2vs = post_word_base(DiscoRnnTree, model_config, data_config, readers)
    from data.cross.binary import BxDM
    get_dm = lambda num_threads: BxDM(disco.batch_size << 1, i2vs, num_threads)
    return DiscoOperator(model, get_datasets, recorder, i2vs, get_dm, train_config, recorder.evalb)