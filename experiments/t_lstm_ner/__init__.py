from data.ner import NerReader, data_type
from data.ner_types import C_ABSTRACT, C_IN
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key, change_key

from experiments.t_lstm_ner.model import LstmNer, model_type
from experiments.t_lstm_ner.operator import NerOperator, train_type
from data.backend import CharTextHelper

get_any_ner = lambda idner = None: idner
def get_configs(recorder = None):
    if recorder is None:
        return {C_IN: data_type}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    ner_data = HParams(get_any_ner(**data_config))
    ner_model = HParams(model_config)

    reader = NerReader(ner_data.data_path,
                       ner_data.source_path,
                       ner_data.with_bi_prefix,
                       ner_data.with_pos_tag,
                       ner_data.vocab_size,
                       CharTextHelper)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN: # TODO clean & noise; ft;
            datasets[C_ABSTRACT] = reader.batch(M_TRAIN,
                                                ner_data.batch_size,
                                                ner_data.bucket_len,
                                                ner_data.ner_extension,
                                                max_len = ner_data.max_len,
                                                sort_by_length = ner_data.sort_by_length)
        else:
            datasets[C_ABSTRACT] = reader.batch(mode, ner_data.batch_size << 1, 0)
        return datasets

    task_params = []
    if ner_model.use.char_rnn:
        task_params.append('num_chars')
    if ner_model.use.word_emb:
        task_params.append('num_tokens')
        task_params.append('initial_weights')
    if ner_data.with_pos_tag:
        task_params.append('num_poses')
    task_params.append('num_bios' if ner_data.with_bi_prefix else 'num_ners')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}
    if ner_model.use.word_emb:
        change_key(task_params, 'num_tokens', 'num_words')
    
    model = LstmNer(**model_config, **task_params)
    model.to(reader.device)
    return NerOperator(model, get_datasets, recorder, reader.i2vs, train_config)
