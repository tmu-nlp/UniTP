from data.penn_types import K_CORP, C_PTB, accp_data_config

CORPORA = {C_PTB}

def get_configs(recorder = None):
    from experiments.t_plm_cm.model import model_type
    from experiments.t_cm.operator import train_type
    if recorder is None:
        _accp_data_config = accp_data_config.copy()
        corps = _accp_data_config.pop(K_CORP)
        _accp_data_config[K_CORP] = {C_PTB: corps[C_PTB]}
        return _accp_data_config, model_type, train_type

    from models.plm import XLNetDatasetHelper
    from experiments.t_cm import make_instance
    return make_instance(recorder, XLNetDatasetHelper)