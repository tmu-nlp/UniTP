from models.xccp import BaseRnnTree, model_type
from utils.types import word_dim
from models.plm import plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim

class DiscoPlmTree(BaseRnnTree):
    def __init__(self,
                 cls_plm,
                 model_dim,
                 paddings,
                 input_layer,
                 **base_config):
        super().__init__(model_dim, **base_config)
        self._input_layer = cls_plm(model_dim, paddings = paddings, **input_layer)

    def forward(self,
                word_idx,
                tune_pre_trained,
                plm_idx, plm_start, **kw_args):
        batch_size, batch_len, base_inputs, bottom_existence = self._input_layer(word_idx, None, plm_idx, plm_start, tune_pre_trained)
        base_returns = super().forward(base_inputs, bottom_existence.squeeze(dim = 2), **kw_args)
        return (batch_size, batch_len, base_inputs, None) + base_returns