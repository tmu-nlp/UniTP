from data.stutt_types import E_DISCONTINUOUS, dccp_data_config

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    from experiments.t_plm_db.model import model_type
    from experiments.t_db.operator import train_type
    if recorder is None:
        return dccp_data_config, model_type, train_type
    from experiments.t_db import make_instance
    return make_instance(recorder, True)