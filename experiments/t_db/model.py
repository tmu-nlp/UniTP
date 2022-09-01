from models.backend import InputLayer, input_config, contextual_config
from models.backend import char_rnn_config
from models.dccp import BaseRnnParser, model_type
from utils.types import word_dim, true_type, false_type
from models.combine import combine_static_type

model_type = model_type.copy()
model_type['model_dim']        = word_dim
model_type['char_rnn']         = char_rnn_config
model_type['word_emb']         = input_config
model_type['use']              = dict(char_rnn = false_type, word_emb = true_type)
model_type['contextual_layer'] = contextual_config
model_type['combine_static']   = combine_static_type

class DiscoRnnTree(InputLayer, BaseRnnParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, squeeze_existence = True, small_endian_tags = True)