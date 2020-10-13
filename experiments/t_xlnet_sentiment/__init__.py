from experiments.t_lstm_sentiment import *
from experiments.t_xlnet_sentiment.model import StanXLNetTree, model_type
from models.plm import XLNetDatasetHelper

def get_configs(recorder = None):
    if recorder is None:
        return {C_SSTB: data_type, C_PTB: parsing_config}, model_type, train_type
        
    data_config, model_config, train_config, _ = recorder.task_specs()

    stan = HParams(data_config[C_SSTB])
    if stan.trapezoid_height:
        trapezoid_specs = stan.trapezoid_height, stan.source_path
    else:
        trapezoid_specs = None

    stan_reader = StanReader(stan.data_path,
                             stan.vocab_size,
                             stan.nil_as_pads,
                             stan.nil_is_neutral,
                             trapezoid_specs,
                             extra_text_helper = XLNetDatasetHelper)
    

    if model_config['tag_label_layer']['hidden_dim']:
        penn = HParams(data_config[C_PTB], fallback_to_none = True)
        train_cnf     = penn.binarization._nested
        non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}

        trapezoid_specs = None
        if penn.trapezoid_height:
            specs = select_and_split_corpus(C_PTB, 
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set)
            data_splits = {k:v for k,v in zip((M_TRAIN, M_DEVEL, M_TEST), specs[-1])}
            trapezoid_specs = specs[:-1] + (data_splits, penn.trapezoid_height)

        penn_reader = PennReader(penn.data_path,
                                 penn.vocab_size,
                                 True, # load_label
                                 penn.unify_sub,
                                 penn.with_ftags,
                                 penn.nil_as_pads,
                                 trapezoid_specs,
                                 extra_text_helper = XLNetDatasetHelper)

        stan_reader.extend_vocab(penn_reader.i2vs.token, penn_reader.get_to_model('initial_weights'))
        task_params = {'num_polars': stan_reader.get_to_model('num_polars')}
        for pname in ('num_tags', 'num_labels', 'paddings'):
            task_params[pname] = penn_reader.get_to_model(pname)
        penn_pad, stan_pad = (len(r.get_to_model('paddings')) for r in (penn_reader, stan_reader))
        assert penn_pad == stan_pad
        penn_i2vs = penn_reader.i2vs
        stan_i2vs = stan_reader.i2vs
        stan_i2vs.create(token = penn_i2vs.token)
    else:
        stan_i2vs = penn_i2vs = stan_reader.i2vs
        task_params = 'num_polars', 'paddings'
        task_params = {pname: stan_reader.get_to_model(pname) for pname in task_params}
        task_params['num_tags'] = task_params['num_labels'] = penn_reader = None
        
    
    def get_datasets(mode):
        
        datasets = {}
        if mode == M_TRAIN:
            datasets[C_SSTB] = stan_reader.batch(M_TRAIN, stan.batch_size, stan.bucket_len,
                                                max_len = stan.max_len, sort_by_length = stan.sort_by_length)
        else:
            datasets[C_SSTB] = stan_reader.batch(mode, stan.batch_size, 0, max_len = stan.max_len)

        if penn_reader is not None:
            if mode == M_TRAIN:
                datasets[C_PTB] = penn_reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, train_cnf,
                                                    max_len = penn.max_len, sort_by_length = penn.sort_by_length)
            else:
                datasets[C_PTB] = penn_reader.batch(mode, penn.batch_size, 0, non_train_cnf, max_len = penn.max_len)
        return datasets

    model = StanXLNetTree(**model_config, **task_params)
    model.to(stan_reader.device)
    return StanOperator(model, get_datasets, recorder, penn_i2vs, stan_i2vs, recorder.evalb, train_config)