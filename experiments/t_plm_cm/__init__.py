from data.penn_types import E_CONTINUE, C_PTB, accp_data_config, select_and_split_corpus
from utils.types import M_TRAIN
from utils.param_ops import HParams, get_sole_key

from models.plm import XLNetDatasetHelper
from experiments.t_plm_cm.model import ContinuousXLNetTree, model_type
from experiments.t_cm.operator import CMOperator, train_type
from experiments.t_plm_cb import get_any_penn

CORPORA = set(E_CONTINUE)

def get_configs(recorder = None):
    if recorder is None:
        return accp_data_config, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)

    from data.penn import MultiReader
    (corpus_reader, get_fnames, _,
     data_splits) = select_and_split_corpus(get_sole_key(data_config), 
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set)

    reader = MultiReader(penn.data_path,
                         penn.balanced > 0,
                         penn.unify_sub,
                         corpus_reader,
                         get_fnames,
                         data_splits,
                         penn.vocab_size,
                         False,
                         XLNetDatasetHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_PTB] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                           balanced = penn.balanced,
                                           max_len = penn.max_len,
                                           sort_by_length = penn.sort_by_length)
        else:
            datasets[C_PTB] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}

    model = ContinuousXLNetTree(**model_config, **task_params)
    return CMOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb, train_config)