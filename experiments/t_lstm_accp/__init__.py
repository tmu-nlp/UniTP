from data.penn import MAryReader
from data.penn_types import C_ABSTRACT, accp_data_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key
from utils.shell_io import byte_style

from experiments.t_lstm_accp.model import MaryRnnTree, model_type
from experiments.t_lstm_accp.operator import PennOperator, train_type

get_any_penn = lambda ptb = None, ctb = None, ktb = None: ptb or ctb or ktb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: accp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)

    (corpus_reader, get_fnames, _,
     data_splits) = select_and_split_corpus(get_sole_key(data_config), 
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set)

    reader = MAryReader(penn.data_path,
                        penn.unify_sub,
                        corpus_reader,
                        get_fnames,
                        data_splits,
                        penn.vocab_size)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                                max_len = penn.max_len,
                                                sort_by_length = penn.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, penn.batch_size, 0)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('initial_weights', 'num_tokens', 'num_tags', 'num_labels', 'paddings')}

    model = MaryRnnTree(**model_config, **task_params)
    model.to(reader.device)
    return PennOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb, train_config)
        
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

