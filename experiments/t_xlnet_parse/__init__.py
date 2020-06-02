from data.penn import PennReader
from data.penn_types import C_ABSTRACT, parsing_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST, BaseType, frac_open_0
from utils.param_ops import HParams, get_sole_key

from experiments.t_xlnet_parse.model import XLNetPennTree, xlnet_penn_tree_config, XLNetDatasetHelper
from experiments.t_lstm_parse.operator import PennOperator, train_type

require_source_path = False
train_type = train_type.copy()
train_type['learning_rate'] = BaseType(1e-5, validator = frac_open_0)

get_any_penn = lambda ptb = None, ctb = None: ptb or ctb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: parsing_config}, xlnet_penn_tree_config, train_type
    
    data_config, model_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config))
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
                        trapezoid_specs,
                        extra_text_helper = XLNetDatasetHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, train_cnf,
                                                max_len = penn.max_len, sort_by_length = penn.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, penn.batch_size, 0, non_train_cnf, max_len = 100)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}

    model = XLNetPennTree(**model_config, **task_params)
    model.to(reader.device)
    return  PennOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb, train_type)