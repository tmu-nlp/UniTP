from models.backend import InputLayer, input_config, contextual_config
from models.combine import combine_static_type
from models.accp import _CM, model_type
from utils.types import word_dim

model_type = model_type.copy()
model_type['model_dim']        = word_dim
model_type['input_emb']        = input_config
model_type['contextualize'] = contextual_config
model_type['combine_emb_and_cxt']   = combine_static_type

class CM(InputLayer, _CM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, squeeze_existence = True, small_endian_tags = True)