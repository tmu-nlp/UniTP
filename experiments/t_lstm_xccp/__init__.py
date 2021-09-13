from data.disco import DiscoMultiReader
from data.disco_types import C_ABSTRACT, C_DPTB, C_TGR, xccp_data_config, select_and_split_corpus, parpath
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key
from data.backend import CharTextHelper

from experiments.t_lstm_dccp import get_any_disco
from experiments.t_lstm_xccp.model import DiscoMultiRnnTree, model_type
from experiments.t_lstm_xccp.operator import DiscoMultiOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: xccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_disco(**data_config), fallback_to_none = True)
    corp_name = get_sole_key(data_config)

    model = HParams(model_config)
    data_splits = select_and_split_corpus(corp_name,
                                          penn.source_path,
                                          penn.data_splits.train_set,
                                          penn.data_splits.devel_set,
                                          penn.data_splits.test_set,
                                          binary = False,
                                          read_dep = parpath(penn.data_path))

    reader = DiscoMultiReader(penn.data_path,
                              penn.medium_factor.balanced > 0,
                              penn.unify_sub,
                              penn.continuous_fence_only,
                              data_splits,
                              penn.vocab_size,
                              None,
                              CharTextHelper if model.use.char_rnn else None)
    
    def get_datasets(mode, new_medium_factor = None):
        datasets = {}
        if mode == M_TRAIN:
            train_ds = reader.loaded_ds.get(mode)
            if train_ds is None:
                datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                                   penn.medium_factor._nested,
                                                   max_len = penn.max_len,
                                                   min_gap = penn.min_gap,
                                                   sort_by_length = penn.sort_by_length)
            else:
                from data.backend import post_batch
                train_ds.reset_factors(new_medium_factor)
                datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
        else:
            datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = ['num_tags', 'num_labels', 'paddings']
    if model.use.word_emb:
        task_params += ['initial_weights', 'num_tokens']
    if model.use.char_rnn:
        task_params.append('num_chars')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}
    
    model_config['space_layer']['continuous_fence_only'] = penn.continuous_fence_only
    model = DiscoMultiRnnTree(**model_config, **task_params)
    model.to(reader.device)
    # train_config.create(label_log_freq_inv = reader.frequency('label', log_inv = True))
    from data.cross.multib import MxDM
    get_dm = lambda i2vs, num_threads: MxDM(penn.batch_size << 1, i2vs, num_threads)
    return DiscoMultiOperator(model, get_datasets, recorder, reader.i2vs, get_dm, train_config, recorder.evalb)
