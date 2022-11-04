from data.stutt_types import xccp_data_config, E_DISCONTINUOUS, C_TIGER, C_DPTB

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    from experiments.t_dm.model import model_type
    from experiments.t_dm.operator import train_type
    if recorder is None:
        return xccp_data_config, model_type, train_type
    return make_instance(recorder)
    
def make_instance(recorder, use_plm = False):
    from data.stutt import DTreeReader
    from utils.types import M_TRAIN, K_CORP
    from utils.param_ops import HParams
    from data.utils import post_word_base, parsing_pnames

    data_config, model_config, train_config, _ = recorder.task_specs()
    stutt = HParams(data_config)
    readers = {}

    if use_plm:
        if len(data_config[K_CORP]) == 1:
            from models.xccp import _DM
            from models.plm import XLNetDatasetHelper, XLNetLeaves, GBertDatasetHelper, GBertLeaves
            if C_TIGER in data_config[K_CORP]:
                DatasetHelper = GBertDatasetHelper
                class DM(GBertLeaves, _DM):
                    def forward(self, *args, **kw_args):
                        return super().forward(*args, **kw_args, squeeze_existence = True)

            elif C_DPTB in data_config[K_CORP]:
                DatasetHelper = XLNetDatasetHelper
                class DM(XLNetLeaves, _DM):
                    def forward(self, *args, **kw_args):
                        return super().forward(*args, **kw_args, squeeze_existence = True)
            else:
                raise ValueError(', '.join(data_config[K_CORP].keys()))
        else:
            raise ValueError(', '.join(data_config[K_CORP].keys()))
        from experiments.t_plm_dm.operator import DMOperater_lr as DMOperater
        pnames = parsing_pnames[2:]
    else:
        from experiments.t_dm.operator import DMOperater
        from experiments.t_dm.model import DM
        DatasetHelper = None
        pnames = parsing_pnames
    
    for corp_name, dc in data_config[K_CORP].items():
        dc['token'] = 'word'
        dc = HParams(dc)
        readers[corp_name] = DTreeReader(corp_name, dc, stutt.unify_sub, extra_text_helper = DatasetHelper)
        if dc.disco_2d.inter_rate > 0 and dc.max_interply > 0:
            assert train_config.loss_weight.disco_2d_inter > 0
        if dc.disco_2d.intra_rate > 0:
            assert train_config.loss_weight.disco_2d_intra > 0
    
    def get_datasets(mode, new_factor = None):
        datasets = {}
        for corp_name, reader in readers.items():
            if mode == M_TRAIN:
                train_ds = reader.loaded_ds.get(mode)
                if train_ds is None:
                    datasets[corp_name] = reader.multib(
                        M_TRAIN,
                        stutt.batch_size,
                        stutt.bucket_len,
                        stutt.continuous_chunk_only,
                        max_len = stutt.max_len,
                        sort_by_length = stutt.sort_by_length,
                        new_factor = new_factor[corp_name] if new_factor else None)
                else:
                    from data.dataset import post_batch
                    train_ds.reset_multib_factor(*new_factor[corp_name])
                    datasets[corp_name] = post_batch(mode, train_ds, stutt.sort_by_length, stutt.bucket_len, stutt.batch_size)
            else:
                datasets[corp_name] = reader.multib(mode, stutt.batch_size << 1, 0, stutt.continuous_chunk_only)
        return datasets

    model_config['space_layer']['continuous_chunk_only'] = stutt.continuous_chunk_only
    model, i2vs = post_word_base(DM, model_config, data_config[K_CORP], readers, pnames)
    from data.cross.mp import MxDM
    get_dm = lambda num_threads: MxDM(stutt.batch_size << 1, i2vs, num_threads)
    return DMOperater(model, get_datasets, recorder, i2vs, get_dm, train_config, recorder.evalb)
