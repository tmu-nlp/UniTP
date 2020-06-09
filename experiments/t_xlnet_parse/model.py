from models.nccp import BaseRnnTree, penn_tree_config, torch, nn, Tensor
from utils.types import word_dim, num_ctx_layer, frac_2, frac_4
from models.xlnet import subword_proc, XLNetLeaves

from models.backend import contextual_type, activation_type
xlnt_leaves_config = dict(contextual   = contextual_type,
                          num_layers   = num_ctx_layer,
                          drop_out     = frac_4,
                          rnn_drop_out = frac_2,
                          activation   = activation_type,
                          subword_proc = subword_proc)
model_type = penn_tree_config.copy()
model_type['input_layer'] = xlnt_leaves_config
model_type['model_dim']   = word_dim

class XLNetPennTree(BaseRnnTree):
    def __init__(self,
                 model_dim,
                 paddings,
                 input_layer,
                 **base_config):
        super().__init__(model_dim, **base_config)
        self._input_layer = XLNetLeaves(model_dim, paddings = paddings, **input_layer)
        self._paddings    = paddings

    def forward(self,
                word_idx,
                tune_pre_trained,
                offset, xl_ids, xl_start, 
                ingore_logits = False, **kw_args):
        batch_size, batch_len = word_idx.shape
        static, dynamic = self._input_layer(word_idx, offset, xl_ids, xl_start, tune_pre_trained)

        if self._paddings:
            bottom_existence = torch.ones(batch_size, batch_len,
                                          dtype  = torch.bool,
                                          device = word_idx.device)
        else:
            bottom_existence = word_idx > 0
        base_returns = super().forward(dynamic, bottom_existence, ingore_logits, **kw_args)
        return (batch_size, batch_len, static, dynamic, None) + base_returns