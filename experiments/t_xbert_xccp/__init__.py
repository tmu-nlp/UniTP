
from data.disco import DiscoMultiReader
from data.disco_types import C_ABSTRACT, xccp_data_config, select_and_split_corpus
from utils.types import M_TRAIN
from utils.param_ops import HParams

from experiments.t_xbert_dccp import get_any_disco
from experiments.t_xbert_xccp.model import DiscoPlmTree, model_type
from experiments.t_lstm_xccp.operator import DiscoMultiOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: xccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    corp_name, penn, DatasetHelper, Leaves = get_any_disco(**data_config)
    penn = HParams(penn, fallback_to_none = True)

    model = HParams(model_config)
    data_splits = select_and_split_corpus(corp_name,
                                          penn.source_path,
                                          penn.data_splits.train_set,
                                          penn.data_splits.devel_set,
                                          penn.data_splits.test_set,
                                          False, False) # TODO dep

    reader = DiscoMultiReader(penn.data_path,
                              penn.medium_factor.balanced > 0,
                              penn.unify_sub,
                              penn.continuous_fence_only,
                              data_splits,
                              penn.vocab_size,
                              None,
                              DatasetHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, penn.medium_factor._nested,
                                               max_len = penn.max_len,
                                               min_gap = penn.min_gap,
                                               sort_by_length = penn.sort_by_length)
        else:
            datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tags', 'num_labels', 'paddings')}
    model_config['space_layer']['continuous_fence_only'] = penn.continuous_fence_only
    model = DiscoPlmTree(Leaves, **model_config, **task_params)
    model.to(reader.device)
    from data.cross.multib import MxDM
    get_dm = lambda i2vs, num_threads: MxDM(penn.batch_size << 1, i2vs, num_threads)
    return DiscoMultiOperator(model, get_datasets, recorder, reader.i2vs, get_dm, train_config, recorder.evalb)