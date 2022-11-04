from models.nccp import model_type, _CB, _SentimentCB, _SentimentOnSyntacticCB
from utils.types import word_dim
from models.plm import XLNetLeaves, plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim

class XLNetCB(XLNetLeaves, _CB):
    def __init__(self, *largs, **kwargs):
        super().__init__(*largs, **kwargs)

    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)

class XLNetSentimentCB(XLNetLeaves, _SentimentCB):
    def __init__(self, *largs, **kwargs):
        super().__init__(*largs, **kwargs)

    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)

class XLNetSentimentOnSyntacticCB(XLNetLeaves, _SentimentOnSyntacticCB):
    def __init__(self, *largs, **kwargs):
        super().__init__(*largs, **kwargs)

    def forward(self, *largs, **kwargs):
        return super().forward(*largs, **kwargs, squeeze_existence = False)