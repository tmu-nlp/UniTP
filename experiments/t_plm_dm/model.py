from models.xccp import model_type
from utils.types import word_dim
from models.plm import plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim

from models.xccp import _DM
from models.plm import XLNetLeaves, GBertLeaves

class GBertDM(GBertLeaves, _DM):
    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, offset = None, squeeze_existence = True)

class XLNetDM(XLNetLeaves, _DM):
    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, offset = None, squeeze_existence = True)