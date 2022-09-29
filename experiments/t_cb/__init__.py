from data.penn_types import E_BINARY, nccp_data_config, C_SSTB

CORPORA = set(E_BINARY)

def get_configs(recorder = None):
    from experiments.t_cb.model import model_type
    from experiments.t_cb.operator import CBOperator, train_type
    if recorder is None:
        return nccp_data_config, model_type, train_type
    
    from data.penn import PennReader
    from data.stan import StanReader
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams
    
    data_config, model_config, train_config, _ = recorder.task_specs(True)
    readers = {}
    penn = HParams(data_config)
    has_penn = has_sstb = False
    
    for corp_name, dc in data_config[K_CORP].items():
        if corp_name == C_SSTB:
            has_sstb = True
            reader = StanReader(HParams(dc))
        else:
            has_penn = True
            reader = PennReader(corp_name,
                HParams(dc, fallback_to_none = True),
                penn.unify_sub,
                penn.nil_pad)
        readers[corp_name] = reader

    if not has_penn:
        model_config.pop('tag_label_layer', None)
    if not has_sstb:
        model_config.pop('polar_layer', None)
                
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                extra = {}
                if new_factor and corp_name != C_SSTB:
                    extra['new_factor'] = new_factor[corp_name]
                if (train_ds := reader.loaded_ds.get(mode)) is None:
                    datasets[corp_name] = reader.binary(
                        M_TRAIN,
                        penn.condense_per,
                        penn.batch_size,
                        penn.bucket_len, 
                        max_len = penn.max_len,
                        sort_by_length = penn.sort_by_length,
                        **extra)
                else:
                    if corp_name != C_SSTB:
                        from data.dataset import post_batch
                        train_ds.reset_binary_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, penn.sort_by_length, penn.bucket_len, penn.batch_size)
            else:
                datasets[corp_name] = reader.binary(mode, penn.condense_per, penn.batch_size << 1, 0)
        return datasets

    from data.utils import post_word_base, parsing_pnames, sentiment_pnames, parsing_sentiment_pnames
    if has_sstb:
        if has_penn:
            from experiments.t_cb.model import SentimentOnSyntacticCB as CB
            pnames = parsing_sentiment_pnames
        else:
            from experiments.t_cb.model import SentimentCB as CB
            pnames = sentiment_pnames
    else:
        from experiments.t_cb.model import CB
        pnames = parsing_pnames

    if has_penn:
        if not penn.condense_per:
            from data.continuous.binary.mp import TriangularDM as dm_cls
        else:
            from data.continuous.binary.mp import TrapezoidalDM as dm_cls
        get_dm = lambda num_threads: dm_cls(penn.batch_size << 1, i2vs, num_threads)
    else:
        get_dm = None

    model, i2vs = post_word_base(CB, model_config, data_config[K_CORP], readers, pnames)
    return CBOperator(model, get_datasets, recorder, i2vs, get_dm, recorder.evalb, train_config)