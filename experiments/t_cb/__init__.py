from data.penn import PennReader
from data.penn_types import C_ABSTRACT, C_KTB, nccp_data_config
from data.penn_types import select_and_split_corpus, select_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams
from utils.shell_io import byte_style
from data.backend import pre_word_base, post_word_base

from experiments.t_cb.model import ContinuousRnnTree, model_type
from experiments.t_cb.operator import PennOperator, train_type

def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: nccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    chelper = pre_word_base(model_config)
    train_cnf, non_train_cnf = {}, {}
    for corp_name in data_config:
        penn = HParams(data_config[corp_name], fallback_to_none = True)
        if penn.trapezoid_height:
            specs = select_and_split_corpus(corp_name,
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set)
            data_splits = {k:v for k,v in zip((M_TRAIN, M_DEVEL, M_TEST), specs[-1])}
            trapezoid_specs = specs[:-1] + (data_splits, penn.trapezoid_height, corp_name == C_KTB)
            from data.continuous.binary.trapezoid.dataset import TrapezoidalDM as dm_cls
            prompt = f'Use trapezoidal data (stratifying height: {penn.trapezoid_height})', '2'
        else:
            trapezoid_specs = None
            from data.continuous.binary.triangle.dataset import TriangularDM as dm_cls
            prompt = f'Use triangular data (stratifying height: +inf)', '3'
        
        readers[corp_name] = PennReader(
            penn.data_path,
            penn.vocab_size,
            True, # load_label
            penn.unify_sub,
            penn.with_ftags,
            penn.nil_as_pads,
            trapezoid_specs,
            chelper)
        train_cnf[corp_name] = cnf = penn.binarization._nested
        non_train_cnf[corp_name] = {max(cnf, key = cnf.get): 1}
    print(byte_style(*prompt))
    
    def get_datasets(mode):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                datasets[corp_name] = reader.batch(
                    M_TRAIN,
                    penn.batch_size,
                    penn.bucket_len,
                    train_cnf[corp_name],
                    max_len = penn.max_len,
                    sort_by_length = penn.sort_by_length)
            else:
                datasets[corp_name] = reader.batch(
                    mode,
                    penn.batch_size << 1, 0,
                    non_train_cnf[corp_name])
        return datasets

    model, i2vs = post_word_base(ContinuousRnnTree, model_config, data_config, readers)
    get_dm = lambda num_threads: dm_cls(penn.batch_size << 1, i2vs, num_threads)
    return PennOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)