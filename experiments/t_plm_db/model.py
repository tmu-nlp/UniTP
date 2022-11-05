from models.dccp import model_type
from utils.types import word_dim
from models.plm import plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim


from models.dccp import _DB
from models.plm import XLNetLeaves, GBertLeaves
class GBertDB(GBertLeaves, _DB):
    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, offset = 1, squeeze_existence = True)

class XLNetDB(XLNetLeaves, _DB):
    def forward(self, *args, **kw_args):
        return super().forward(*args, **kw_args, offset = 1, squeeze_existence = True)