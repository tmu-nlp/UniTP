from data.disco import DiscoReader
from data.disco_types import C_ABSTRACT, dccp_data_config
from utils.types import M_TRAIN
from utils.param_ops import HParams, get_sole_key
from data.backend import CharTextHelper

from experiments.t_lstm_dccp.model import DiscoRnnTree, model_type
from experiments.t_lstm_dccp.operator import DiscoOperator, train_type

get_any_disco = lambda dptb = None, tiger = None: dptb or tiger
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: dccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    disco = HParams(get_any_disco(**data_config), fallback_to_none = True)
    train_cnf     = disco.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}
    corp_name = get_sole_key(data_config)

    model = HParams(model_config)
    reader = DiscoReader(disco.data_path, disco.vocab_size, disco.unify_sub, CharTextHelper if model.use.char_rnn else None)
    
    def get_datasets(mode, new_train_cnf = None):
        datasets = {}
        if mode == M_TRAIN:
            datasets[corp_name] = reader.batch(M_TRAIN, disco.batch_size, disco.bucket_len, new_train_cnf or train_cnf,
                                                shuffle_swap = disco.shuffle_swap,
                                                max_len = disco.max_len,
                                                min_gap = disco.min_gap,
                                                sort_by_length = disco.sort_by_length)
        else:
            datasets[corp_name] = reader.batch(mode, disco.batch_size, 0, non_train_cnf)
        return datasets

    task_params = ['num_tags', 'num_labels', 'paddings']
    if model.use.word_emb:
        task_params += ['initial_weights', 'num_tokens']
    if model.use.char_rnn:
        task_params.append('num_chars')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}

    model = DiscoRnnTree(**model_config, **task_params)
    model.to(reader.device)
    train_config.create(label_log_freq_inv = reader.frequency('label', log_inv = True))
    return DiscoOperator(model, get_datasets, recorder, reader.i2vs, train_config, recorder.evalb)