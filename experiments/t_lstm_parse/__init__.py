from data.penn import PennReader
from data.penn_types import C_ABSTRACT, parsing_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key

from experiments.t_lstm_parse.model import LstmPennTree, lstm_penn_tree_config
from experiments.t_lstm_parse.operator import PennOperator

require_source_path = False

get_any_penn = lambda ptb = None, ctb = None, ktb = None: ptb or ctb or ktb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: parsing_config}, lstm_penn_tree_config
    
    data_config, model_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)
    train_cnf     = penn.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}

    trapezoid_specs = None
    if penn.trapezoid_height:
        specs = select_and_split_corpus(get_sole_key(data_config), 
                                        penn.source_path,
                                        penn.data_splits.train_set,
                                        penn.data_splits.devel_set,
                                        penn.data_splits.test_set)
        data_splits = {k:v for k,v in zip((M_TRAIN, M_DEVEL, M_TEST), specs[-1])}
        trapezoid_specs = specs[:-1] + (data_splits, penn.trapezoid_height)

    reader = PennReader(penn.data_path,
                        penn.vocab_size,
                        True, # load_label
                        penn.unify_sub,
                        penn.with_ftags,
                        penn.nil_as_pads,
                        trapezoid_specs)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, train_cnf,
                                                max_len = penn.max_len, sort_by_length = penn.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, penn.batch_size, 0, non_train_cnf, max_len = 100)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('initial_weights', 'num_words', 'num_tags', 'num_labels', 'paddings')}

    model = LstmPennTree(**model_config, **task_params)
    model.to(reader.device)
    return  PennOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb)
        
# def get_datasets_for_tagging(ptb = None, ctb = None, ktb = None):
#     if not (ptb or ctb or ktb):
#         return dict(penn = none_type)

#     datasets = {}
#     penn = ptb or ctb or ktb
#     reader = PennReader(penn['data_path'], False)
    
#     if M_TRAIN in mode_keys:
#         datasets[M_TRAIN] = reader.batch(M_TRAIN, 100, 20, max_len = 100)
#     if M_DEVEL in mode_keys:
#         datasets[M_DEVEL]  = reader.batch(M_DEVEL, 60, 20, max_len = 100)
#     if M_TEST in mode_keys:
#         datasets[M_TEST]  = reader.batch(M_TEST, 60, 20, max_len = 100)
#     return datasets, reader

