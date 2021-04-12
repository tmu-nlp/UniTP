from data.disco import DiscoMultiReader
from data.disco_types import C_ABSTRACT, C_DPTB, C_TGR, xccp_data_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key

from experiments.t_lstm_dccp import get_any_disco
from experiments.t_lstm_xccp.model import DiscoMultiRnnTree, model_type
from experiments.t_lstm_xccp.operator import DiscoMultiOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: xccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_disco(**data_config), fallback_to_none = True)

    data_splits = select_and_split_corpus(get_sole_key(data_config), 
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set,
                                            False, False) # TODO dep

    reader = DiscoMultiReader(penn.data_path,
                              penn.medium_factor.balanced > 0,
                              penn.unify_sub,
                              data_splits,
                              penn.vocab_size)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, penn.medium_factor._nested,
                                                max_len = penn.max_len,
                                                min_gap = penn.min_gap,
                                                sort_by_length = penn.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('initial_weights', 'num_tokens', 'num_tags', 'num_labels', 'paddings')}
    model = DiscoMultiRnnTree(**model_config, **task_params)
    model.to(reader.device)
    train_config.create(label_log_freq_inv = reader.frequency('label', log_inv = True))
    return DiscoMultiOperator(model, get_datasets, recorder, reader.i2vs, train_config, recorder.evalb)