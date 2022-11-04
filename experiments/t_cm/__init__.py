from data.penn_types import E_MULTIB, accp_data_config, E_CONTINUE, E_NER
from experiments.t_cm.operator import CMOperator, train_type

CORPORA = set(E_MULTIB)

def get_configs(recorder = None):
    from experiments.t_cm.model import model_type
    if recorder is None:
        return accp_data_config, model_type, train_type
    return make_instance(recorder)

def make_instance(recorder, xlnet_helper = None):
    
    from data.penn import PennReader
    from data.ner import NerReader
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams

    data_config, model_config, train_config, _ = recorder.task_specs()
    readers = {}
    penn = HParams(data_config)
    any_ner_corpus = any_parsing_corpus = any_char_as_token = False
    
    for corp_name, dc in data_config[K_CORP].items():
        corp_spec = HParams(dc)
        if corp_name in E_CONTINUE:
            readers[corp_name] = PennReader(
                corp_name,
                corp_spec,
                penn.unify_sub,
                extra_text_helper = xlnet_helper)
            if corp_spec.token == 'char':
                any_char_as_token = True
            any_parsing_corpus = True
        elif corp_name in E_NER:
            readers[corp_name] = NerReader(corp_spec, 
            penn.ner.with_bi_prefix,
            extra_text_helper = xlnet_helper)
            any_ner_corpus = True

    if any_char_as_token:
        assert not any_ner_corpus
        from data.continuous import Signal
        Signal.set_char()
    model_config['chunk_layer']['char_chunk'] = any_char_as_token
    
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if corp_name in E_CONTINUE:
                if mode == M_TRAIN:
                    if (train_ds := reader.loaded_ds.get(mode)) is None:
                        datasets[corp_name] = reader.multib(
                            M_TRAIN,
                            penn.batch_size,
                            penn.bucket_len,
                            max_len = penn.max_len,
                            sort_by_length = penn.sort_by_length,
                            new_factor = new_factor[corp_name] if new_factor else None)
                    else:
                        from data.dataset import post_batch
                        train_ds.reset_multib_factor(*new_factor[corp_name])
                        datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
                else:
                    datasets[corp_name] = reader.multib(mode, penn.batch_size << 1, 0)
            elif corp_name in E_NER:
                if mode == M_TRAIN:
                    datasets[corp_name] = reader.batch(
                        M_TRAIN,
                        penn.batch_size,
                        penn.bucket_len,
                        max_len        = penn.max_len,
                        sort_by_length = penn.sort_by_length)
                else:
                    datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    from data.utils import post_word_base, parsing_pnames
    if xlnet_helper is None:
        from experiments.t_cm.model import CM
        pnames = parsing_pnames
    else:
        from experiments.t_plm_cm.model import XLNetCM as CM
        pnames = parsing_pnames[2:]

    model, i2vs = post_word_base(CM, model_config, data_config[K_CORP], readers, pnames)
    if any_parsing_corpus:
        from data.continuous.multib.mp import MultibDM
        get_dm = lambda num_threads: MultibDM(penn.batch_size << 1, i2vs, num_threads)
    else:
        get_dm = None
    return CMOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)