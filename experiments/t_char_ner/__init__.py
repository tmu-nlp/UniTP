from data.ner import NerReader, data_config
from data.ner_types import C_ABSTRACT, C_IN
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key

data_type = dict(data_config)
data_type.pop('vocab_size')

from experiments.t_char_ner.model import CharNer, char_model_type
from experiments.t_char_ner.operator import NerOperator, train_type
from data.backend import CharTextHelper

get_any_ner = lambda idner = None: idner
def get_configs(recorder = None):
    if recorder is None:
        return {C_IN: data_type}, char_model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    ner = HParams(get_any_ner(**data_config))

    reader = NerReader(ner.data_path,
                       ner.source_path,
                       ner.with_bi_prefix,
                       ner.with_pos_tag,
                       None,
                       CharTextHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN: # TODO clean & noise; ft;
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN, ner.batch_size, ner.bucket_len, ner.ner_extension,
                                                max_len = ner.max_len,
                                                sort_by_length = ner.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, ner.batch_size << 1, 0)
        return datasets

    task_params = ['num_chars']
    if ner.with_pos_tag:
        task_params.append('num_poses')
    task_params.append('num_bios' if ner.with_bi_prefix else 'num_ners')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}
    
    model = CharNer(**model_config, **task_params)
    model.to(reader.device)
    return NerOperator(model, get_datasets, recorder, reader.i2vs, train_config)
