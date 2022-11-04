from data.stutt_types import xccp_data_config, E_DISCONTINUOUS

CORPORA = set(E_DISCONTINUOUS)

def get_configs(recorder = None):
    from experiments.t_plm_dm.model import model_type
    from experiments.t_dm.operator import train_type
    if recorder is None:
        return xccp_data_config, model_type, train_type
    from experiments.t_dm import make_instance
    return make_instance(recorder, True)