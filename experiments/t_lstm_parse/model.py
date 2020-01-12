from models.backend import InputLeaves, Contextual, input_config, contextual_config
from models.nccp import BaseRnnTree, penn_tree_config
from utils.types import word_dim

lstm_penn_tree_config = penn_tree_config.copy()
lstm_penn_tree_config['model_dim']        = word_dim
lstm_penn_tree_config['input_layer']      = input_config
lstm_penn_tree_config['contextual_layer'] = contextual_config

class PennRnnTree(BaseRnnTree):
    def __init__(self,
                 num_words,
                 initial_weights,
                 paddings,
                 model_dim,
                 input_layer,
                 contextual_layer,
                 **base_config):
        super().__init__(model_dim, **base_config)
        self._input_layer = InputLeaves(model_dim, num_words, initial_weights, **input_layer)
        self._contextual_layer = Contextual(model_dim, **contextual_layer)

    def forward(self, word_idx, **kw_args):
        batch_size, batch_len    = word_idx.shape
        static, bottom_existence = self._input_layer(word_idx)
        dynamic      = self._contextual_layer(static)
        base_inputs  = static if dynamic is None else dynamic
        base_returns = super().forward(base_inputs, bottom_existence, **kw_args)
        return (batch_size, batch_len, static, dynamic) + base_returns