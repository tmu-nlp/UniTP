from data.penn import PennReader
from data.penn_types import C_ABSTRACT, parsing_config
from utils.types import M_TRAIN
from utils.param_ops import HParams

from experiments.t_xlnet_parse.model import XLNetPennTree, xlnet_penn_tree_config, XLNetDatasetHelper
from experiments.t_lstm_parse.operator import PennOperator

require_source_path = False

get_any_penn = lambda ptb = None, ctb = None: ptb or ctb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: parsing_config}, xlnet_penn_tree_config
    
    data_config, model_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config))
    train_cnf     = penn.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}
    
    reader = PennReader(penn.data_path, penn.vocab_size, True, penn.unify_sub, penn.with_ftags, penn.nil_as_pads, extra_text_helper = XLNetDatasetHelper)
    
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
    return  PennOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb)