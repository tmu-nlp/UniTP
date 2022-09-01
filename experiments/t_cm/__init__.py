from data.penn import MultiReader
from data.penn_types import C_ABSTRACT, C_KTB, accp_data_config
from data.penn_types import select_and_split_corpus, select_corpus
from utils.types import M_TRAIN
from utils.param_ops import HParams
from data.backend import pre_word_base, post_word_base

from experiments.t_cm.model import MultiRnnTree, model_type
from experiments.t_cm.operator import MultiOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: accp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    chelper = pre_word_base(model_config)
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
    
    def get_datasets(mode, balanced = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                if train_ds := reader.loaded_ds.get(mode):
                    from data.backend import post_batch
                    train_ds.reset_factors(balanced[corp_name])
                    datasets[corp_name] = post_batch(
                        mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
                else:
                    datasets[corp_name] = reader.batch(
                        M_TRAIN,
                        penn.batch_size,
                        penn.bucket_len,
                        balanced = balanced[corp_name] if balanced else penn.balanced,
                        max_len = penn.max_len,
                        sort_by_length = penn.sort_by_length)
            else:
                datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    model, i2vs = post_word_base(MultiRnnTree, model_config, data_config, readers)
    from data.multib import MaryDM
    get_dm = lambda num_threads: MaryDM(penn.batch_size << 1, i2vs, num_threads)
    return MultiOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)