from data.penn import LexiconReader, tokenization_config
from data.penn_types import C_ABSTRACT
from utils.types import M_TRAIN, M_DEVEL, M_TEST, frac_7, frac_06, distance_type
from utils.param_ops import HParams, get_sole_key

from experiments.t_lstm_tokenize.model import RnnTokenizer, tokenizer_config
from experiments.t_lstm_tokenize.operator import TokenizerOperator, train_type
from experiments.t_lstm_tokenize.types import D_NOISE, D_CLEAN

def make_noise(ip, op, dst = None):
    if dst is None:
        return dict(outer_p = op, inner_p = ip)
    return dict(outer_p = op, inner_p = ip, distance = dst)

noise_type = dict(origin = frac_7,
                  replace_all = frac_06,
                  swap    = make_noise(frac_06, frac_06, distance_type),
                  replace = make_noise(frac_06, frac_06),
                  insert  = make_noise(frac_06, frac_06),
                  delete  = make_noise(frac_06, frac_06))
tokenization_config = tokenization_config.copy()
tokenization_config['noise'] = noise_type

get_any_penn = lambda ptb = None, ctb = None, ktb = None: ptb or ctb or ktb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: tokenization_config}, tokenizer_config, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)

    noise = {}
    noise_specs = {}
    for name, prob in penn.noise._nested.items():
        if isinstance(prob, float):
            noise[name] = prob
        else:
            noise[name] = prob['outer_p']
            if 'distance' in prob:
                noise_specs[name] = prob['inner_p'], prob['distance']
            else:
                noise_specs[name] = prob['inner_p']

    reader = LexiconReader(penn.data_path,
                           penn.lower_case)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[D_NOISE] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, noise_specs, noise,
                                             max_len = penn.max_len, sort_by_length = penn.sort_by_length)
            datasets[D_CLEAN] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, None, {'origin': 1},
                                             max_len = penn.max_len, sort_by_length = penn.sort_by_length)
        else:
            datasets[D_NOISE] = reader.batch(mode, penn.batch_size, 0, None, {'origin': 1})
            datasets[D_CLEAN] = reader.batch(mode, penn.batch_size, 0, None, {'origin': 1})
        return datasets

    task_params = {pname: reader.get_to_model(pname) for pname in ('num_tokens', 'paddings')}

    model = RnnTokenizer(**model_config, **task_params)
    model.to(reader.device)
    return  TokenizerOperator(model, get_datasets, recorder, reader.i2vs, train_config)