from data.penn_types import C_PTB, C_SSTB, K_CORP, nccp_data_config
CORPORA = {C_PTB, C_SSTB}

def get_configs(recorder = None):
    from experiments.t_plm_cb.model import model_type
    from experiments.t_cb.operator import train_type
    _nccp_data_config = nccp_data_config.copy()
    only_ptb_and_sstb = {k: v for k, v in nccp_data_config[K_CORP].items() if k in CORPORA}
    _nccp_data_config[K_CORP] = only_ptb_and_sstb
    if recorder is None:
        return _nccp_data_config, model_type, train_type
    from models.plm import XLNetDatasetHelper
    from experiments.t_cb import make_instance
    return make_instance(recorder, XLNetDatasetHelper)