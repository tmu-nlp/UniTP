from data.disco import DiscoReader
from data.disco_types import C_ABSTRACT, disco_config
from utils.types import M_TRAIN
from utils.param_ops import HParams
from data.cross.evalb_lcfrs import read_param

from experiments.t_lstm_disco.model import DiscoRnnTree, model_type
from experiments.t_lstm_disco.operator import DiscoOperator, train_type

require_source_path = False

get_any_disco = lambda dptb = None, tiger = None: dptb or tiger
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: disco_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    disco = HParams(get_any_disco(**data_config), fallback_to_none = True)
    train_cnf     = disco.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}

    reader = DiscoReader(disco.data_path, disco.vocab_size, disco.unify_sub)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, disco.batch_size, disco.bucket_len, train_cnf,
                                                shuffle_swap = disco.shuffle_swap, max_len = disco.max_len, sort_by_length = disco.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, disco.batch_size, 0, non_train_cnf)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('initial_weights', 'num_tokens', 'num_tags', 'num_labels', 'paddings')}

    model = DiscoRnnTree(**model_config, **task_params)
    model.to(reader.device)
    return DiscoOperator(model, get_datasets, recorder, reader.i2vs, train_config, read_param(recorder.evalb))