from models.xccp import model_type
from utils.types import word_dim
from models.plm import plm_leaves_config

model_type = model_type.copy()
model_type['input_layer'] = plm_leaves_config
model_type['model_dim']   = word_dim