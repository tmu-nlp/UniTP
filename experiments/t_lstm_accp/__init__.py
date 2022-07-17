from data.penn import MultiReader
from data.penn_types import C_ABSTRACT, C_KTB, accp_data_config
from data.penn_types import select_and_split_corpus, select_corpus
from utils.types import M_TRAIN
from utils.param_ops import HParams, get_sole_key
from data.backend import CharTextHelper

from experiments.t_lstm_accp.model import MultiRnnTree, model_type
from experiments.t_lstm_accp.operator import MultiOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: accp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    model = HParams(model_config)
    chelper = CharTextHelper if model.use.char_rnn else None
    for corp_name in data_config:
        penn = HParams(data_config[corp_name], fallback_to_none = True)
        (corpus_reader, get_fnames, _,
         data_splits) = select_and_split_corpus(corp_name,
                                                penn.source_path,
                                                penn.data_splits.train_set,
                                                penn.data_splits.devel_set,
                                                penn.data_splits.test_set)

        readers[corp_name] = MultiReader(
            penn.data_path,
            penn.balanced > 0,
            penn.unify_sub,
            corpus_reader,
            get_fnames,
            data_splits,
            penn.vocab_size,
            C_KTB == corp_name,
            chelper)
    
    def get_datasets(mode):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                datasets[corp_name] = reader.batch(
                    M_TRAIN,
                    penn.batch_size,
                    penn.bucket_len,
                    balanced = penn.balanced,
                    max_len = penn.max_len,
                    sort_by_length = penn.sort_by_length)
            else:
                datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = ['num_tags', 'num_labels', 'paddings']
    if model.use.word_emb:
        task_params += ['initial_weights', 'num_tokens']
    if model.use.char_rnn:
        task_params.append('num_chars')
    i2vs = {c: r.i2vs for c, r in readers.items()}
    if single_corpus := (len(data_config) == 1):
        single_corpus = get_sole_key(data_config)
        i2vs = i2vs[single_corpus]
    for pname in task_params:
        param = {c: r.get_to_model(pname) for c, r in readers.items()}
        model_config[pname] = param[single_corpus] if single_corpus else param

    model = MultiRnnTree(**model_config)
    from data.multib import MaryDM
    get_dm = lambda i2vs, num_threads: MaryDM(penn.batch_size << 1, i2vs, num_threads)
    return MultiOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)