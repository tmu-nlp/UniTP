from models.accp import _CM, model_type
from utils.types import word_dim
from models.plm import XLNetLeaves, plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim

class XLNetCM(XLNetLeaves, _CM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, squeeze_existence = True)