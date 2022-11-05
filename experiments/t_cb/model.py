from models.backend import InputLayer, input_config, contextual_config
from models.combine import combine_static_type
from models.nccp import model_type, _CB, _SentimentCB, _SentimentOnSyntacticCB
from utils.types import word_dim

model_type = model_type.copy()
model_type['model_dim']      = word_dim
model_type['input_emb']      = input_config
model_type['contextualize']  = contextual_config
model_type['combine_emb_and_cxt'] = combine_static_type

class CB(InputLayer, _CB):
    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)

class SentimentCB(InputLayer, _SentimentCB):
    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)

class SentimentOnSyntacticCB(InputLayer, _SentimentOnSyntacticCB):
    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)