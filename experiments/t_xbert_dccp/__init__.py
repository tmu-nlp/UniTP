from data.disco import DiscoReader
from data.disco_types import C_ABSTRACT, C_DPTB, C_TGR, dccp_data_config
from utils.types import M_TRAIN
from utils.param_ops import HParams, get_sole_key

from experiments.t_xbert_dccp.model import DiscoPlmTree, model_type
from experiments.t_lstm_dccp.operator import DiscoOperator, train_type

def get_any_disco(dptb = None, tiger = None):
    from models.plm import XLNetDatasetHelper, XLNetLeaves, GBertDatasetHelper, GBertLeaves
    if dptb is None:
        return C_TGR, tiger, GBertDatasetHelper, GBertLeaves
    return C_DPTB, dptb, XLNetDatasetHelper, XLNetLeaves

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: dccp_data_config}, model_type, train_type

    data_config, model_config, train_config, _ = recorder.task_specs()
    corp_name, disco, DatasetHelper, Leaves = get_any_disco(**data_config)
    disco = HParams(disco, fallback_to_none = True)
    train_cnf     = disco.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}

    reader = DiscoReader(disco.data_path,
                         disco.vocab_size,
                         disco.unify_sub,
                         extra_text_helper = DatasetHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[corp_name] = reader.batch(M_TRAIN, disco.batch_size, disco.bucket_len, train_cnf,
                                               shuffle_swap = disco.shuffle_swap,
                                               max_len = disco.max_len,
                                               min_gap = disco.min_gap,
                                               sort_by_length = disco.sort_by_length)
        else:
            datasets[corp_name] = reader.batch(mode, disco.batch_size, 0, non_train_cnf)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}

    model = DiscoPlmTree(Leaves, **model_config, **task_params)
    model.to(reader.device)
    from data.cross.binary import BxDM
    get_dm = lambda i2vs, num_threads: BxDM(disco.batch_size << 1, i2vs, num_threads)
    return DiscoOperator(model, get_datasets, recorder, reader.i2vs, get_dm, train_config, recorder.evalb)