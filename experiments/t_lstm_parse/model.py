from models.backend import InputLeaves, Contextual, input_config, contextual_config
from models.nccp import BaseRnnTree, penn_tree_config
from utils.types import word_dim

model_type = penn_tree_config.copy()
model_type['model_dim']        = word_dim
model_type['input_layer']      = input_config
model_type['contextual_layer'] = contextual_config

class PennRnnTree(BaseRnnTree):
    def __init__(self,
                 num_tokens,
                 initial_weights,
                 paddings,
                 model_dim,
                 input_layer,
                 contextual_layer,
                 **base_config):
        super().__init__(model_dim, **base_config)
        self._input_layer = InputLeaves(model_dim, num_tokens, initial_weights, **input_layer)
        self._contextual_layer = Contextual(model_dim, self.hidden_dim, **contextual_layer)

    def forward(self, word_idx, tune_pre_trained, ingore_logits = False, **kw_args):
        batch_size, batch_len    = word_idx.shape
        static, bottom_existence = self._input_layer(word_idx, tune_pre_trained)
        dynamic, final_hidden    = self._contextual_layer(static)
        base_inputs  = static if dynamic is None else dynamic
        base_returns = super().forward(base_inputs, bottom_existence, ingore_logits, **kw_args)
        top3_labels  = super().get_label(final_hidden) if final_hidden is not None else None
        return (batch_size, batch_len, static, dynamic, top3_labels) + base_returns
