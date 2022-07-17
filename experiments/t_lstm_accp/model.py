from models.backend import torch, InputLeaves, Contextual, input_config, contextual_config
from models.backend import PadRNN, char_rnn_config, nn
from models.accp import BaseRnnParser, model_type
from utils.types import word_dim, true_type, false_type
from models.combine import get_combinator, combine_static_type
from torch.nn import ModuleDict

model_type = model_type.copy()
model_type['model_dim']        = word_dim
model_type['char_rnn']         = char_rnn_config
model_type['word_emb']         = input_config
model_type['use']              = dict(char_rnn = false_type, word_emb = true_type)
model_type['contextual_layer'] = contextual_config
model_type['combine_static']   = combine_static_type

class MultiRnnTree(BaseRnnParser):
    def __init__(self,
                 paddings,
                 model_dim,
                 use,
                 word_emb,
                 char_rnn,
                 contextual_layer,
                 combine_static,
                 num_chars       = None,
                 num_tokens      = None,
                 initial_weights = None,
                 **base_config):
        super().__init__(model_dim, **base_config)
        
        if use['word_emb']:
            if isinstance(num_tokens, int):
                self._word_emb = InputLeaves(model_dim, num_tokens, initial_weights, not paddings, **word_emb)
                input_dim = self._word_emb.input_dim
            else:
                from utils.param_ops import get_sole_key
                self._word_emb = ModuleDict({k: InputLeaves(model_dim, v, initial_weights[k], not paddings[k], **word_emb) for k,v in num_tokens.items()})
                input_dim = get_sole_key(n.input_dim for n in self._word_emb.values())
        else:
            self._word_emb = None
            input_dim = model_dim
        if use['char_rnn']:
            embed_dim = char_rnn['embed_dim']
            self._char_rnn = PadRNN(num_chars, None, None, fence_dim = model_dim, char_space_idx = 1, **char_rnn)
            self._char_lin = nn.Linear(embed_dim, model_dim)
            self._char_act = nn.Tanh()
        else:
            self._char_rnn = None

        contextual_layer = Contextual(input_dim, model_dim, self.hidden_dim, **contextual_layer)
        diff = model_dim - input_dim
        self._combine_static = None
        self._bias_only = False
        if contextual_layer.is_useless:
            self._contextual_layer = None
            assert diff == 0, 'useless difference'
        else:
            self._contextual_layer = contextual_layer
            if combine_static:
                self._bias_only = combine_static in ('NS', 'NV')
                self._combine_static = get_combinator(combine_static, input_dim)
            assert diff >= 0, 'invalid difference'
        self._half_dim_diff = diff >> 1

    def forward(self, word_idx, tune_pre_trained,
                sub_idx = None, sub_fence = None, offset = None, key = None,
                **kw_args):
        batch_size, batch_len  = word_idx.shape
        if self._word_emb:
            emb = self._word_emb if key is None else self._word_emb[key]
            static, bottom_existence = emb(word_idx, tune_pre_trained)
            if self._char_rnn:
                char_info = self._char_rnn(sub_idx, sub_fence, offset)
                char_info = self._stem_dp(char_info)
                char_info = self._char_lin(char_info)
                char_info = self._char_act(char_info)
                static = static + char_info * bottom_existence
        else:
            bottom_existence = word_idx > 0
            bottom_existence.unsqueeze_(dim = 2)
            static = self._char_rnn(sub_idx, sub_fence, offset) * bottom_existence
        if self._contextual_layer is None:
            base_inputs = static
            top3_hidden = None
        else:
            dynamic, top3_hidden = self._contextual_layer(static)
            if self._half_dim_diff:
                zero_pads = torch.zeros(batch_size, batch_len, self._half_dim_diff, dtype = static.dtype, device = static.device)
                static = torch.cat([zero_pads, static, zero_pads], dim = 2)
            base_inputs  = dynamic * bottom_existence
            if self._combine_static is not None:
                base_inputs = self._combine_static.compose(static, base_inputs, None)
        base_returns = super().forward(base_inputs, bottom_existence.squeeze(dim = 2), key = key, **kw_args)
        top3_labels  = super().get_label(top3_hidden) if top3_hidden is not None else None
        return (batch_size, batch_len, static, top3_labels) + base_returns

    def state_dict(self, *args, **kwargs):
        odc = super().state_dict(*args, **kwargs)
        emb = self._word_emb
        prefix = '_word_emb.'
        suffix = '_main_emb_layer.weight'
        if isinstance(emb, InputLeaves) and not emb._main_emb_tuned:
            odc.pop(prefix)
        elif isinstance(emb, ModuleDict):
            for k, v in emb.items():
                if not v._main_emb_tuned:
                    odc.pop(prefix + k + '.' + suffix)
        return odc

    def tensorboard(self, recorder, global_step):
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                params = dict(ContextualRatio = ctx_ratio.mean())
                if ctx_ratio.nelement() > 1:
                    params['RatioStdv'] = ctx_ratio.std()
                recorder.tensorboard(global_step, 'Parameters/%s', **params)

    @property
    def message(self):
        # bsb = self._subject_bias.bias
        # msg = f'BlockySoftmax.bias: {bsb.mean()}'
        # if bsb.nelement() > 1:
        #     msg += f'±{bsb.std()}'
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                ctx_ratio *= 100
                msg = 'Contextual Rate:'
                msg += f' {ctx_ratio.mean():.2f}'
                if ctx_ratio.nelement() > 1:
                    msg += f'±{ctx_ratio.std():.2f}%'
                else:
                    msg += '%'
                return msg
