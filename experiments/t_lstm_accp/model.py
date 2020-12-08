from models.backend import torch, InputLeaves, Contextual, input_config, contextual_config, true_type
from models.accp import BaseRnnTree, model_type
from utils.types import word_dim

model_type = model_type.copy()
model_type['model_dim']        = word_dim
model_type['input_layer']      = input_config
model_type['contextual_layer'] = contextual_config
model_type['residual_add']     = true_type

class MaryRnnTree(BaseRnnTree):
    def __init__(self,
                 num_tokens,
                 initial_weights,
                 paddings,
                 model_dim,
                 input_layer,
                 contextual_layer,
                 residual_add,
                 **base_config):
        super().__init__(model_dim, **base_config)
        input_layer = InputLeaves(model_dim, num_tokens, initial_weights, not paddings, **input_layer)
        self._input_layer = input_layer
        input_dim = input_layer.input_dim
        contextual_layer = Contextual(input_dim, model_dim, self.hidden_dim, **contextual_layer)
        diff = model_dim - input_dim
        if contextual_layer.is_useless:
            self._contextual_layer = None
            assert diff == 0, 'useless difference'
        else:
            self._contextual_layer = contextual_layer
            assert diff >= 0, 'invalid difference'
        self._half_dim_diff = diff >> 1
        self._residual_add = residual_add

    def forward(self, word_idx, tune_pre_trained, **kw_args):
        batch_size,   batch_len  = word_idx.shape
        static, bottom_existence = self._input_layer(word_idx, tune_pre_trained)
        if self._contextual_layer is None:
            base_inputs = static
            top3_hidden = None
        else:
            dynamic, top3_hidden = self._contextual_layer(static)
            base_inputs = dynamic * bottom_existence
            if self._half_dim_diff:
                zero_pads = torch.zeros(batch_size, batch_len, self._half_dim_diff, dtype = static.dtype, device = static.device)
                static = torch.cat([zero_pads, static, zero_pads], dim = 2)
            if self._residual_add:
                base_inputs = base_inputs + static
        base_returns = super().forward(base_inputs, bottom_existence.squeeze(dim = 2), **kw_args)
        top3_labels  = super().get_label(top3_hidden) if top3_hidden is not None else None
        return (batch_size, batch_len, static, top3_labels) + base_returns

