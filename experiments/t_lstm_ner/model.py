from models.ner import base_config, torch, RnnNer
from models.backend import char_rnn_config, char_rnn_config, nn
from models.backend import InputLeaves, input_config, PadRNN
from models.types import rnn_module_type
from utils.types import hidden_dim, frac_2, frac_4, true_type, false_type, num_ori_layer, orient_dim
from models.combine import get_combinator, combine_static_type

model_type = dict(char_rnn = char_rnn_config,
                  word_emb = input_config,
                  use = dict(char_rnn = true_type, word_emb = false_type))
model_type.update(base_config)
# model_type['combine_static'] = combine_static_type

class LstmNer(RnnNer):
    def __init__(self,
                 model_dim,
                 use,
                 char_rnn,
                #  combine_static,
                 base_layer,
                 ner_extension,
                 word_emb = None,
                 num_chars = None,
                 num_words = None,
                 num_poses = None,
                 num_bios = None,
                 num_ners = None,
                 initial_weights = None):
        super().__init__(num_poses,
                         num_bios,
                         num_ners,
                         model_dim,
                         ner_extension,
                         **base_layer)
        if use['word_emb']:
            self._word_emb = InputLeaves(model_dim, num_words, initial_weights, True, **word_emb)
        else:
            self._word_emb = None
        if use['char_rnn']:
            self._char_rnn = PadRNN(num_chars, None, None, fence_dim = model_dim, char_space_idx = 1, **char_rnn)
        else:
            self._char_rnn = None
        # if combine_static:
        #     self._bias_only = combine_static in ('NS', 'NV')
        #     self._combine_static = get_combinator(combine_static, input_dim)
        # else:
        #     self._combine_static = None
        #     self._bias_only = False
    
    def forward(self, word_idx, tune_pre_trained = False,
                sub_idx = None, sub_fence = None,
                pos_idx = None, supervised_fence = None, **kw_args):
        batch_size, batch_len = word_idx.shape
        if self._word_emb:
            unit_emb, existence = self._word_emb(word_idx, tune_pre_trained)
            if self._char_rnn:
                unit_emb = unit_emb + self._char_rnn(sub_idx, sub_fence) * existence
            existence = existence.squeeze(dim = 2)
        else:
            existence = word_idx > 0
            unit_emb = self._char_rnn(sub_idx, sub_fence) * existence.unsqueeze(dim = 2)
        return (batch_size, batch_len, existence) + super().forward(unit_emb, existence, pos_idx, supervised_fence)

    # def tensorboard(self, recorder, global_step):
    #     if self._bias_only:
    #         ctx_ratio = self._combine_static.itp_rhs_bias().detach()
    #         if ctx_ratio is not None:
    #             params = dict(ContextualRatio = ctx_ratio.mean())
    #             if ctx_ratio.nelement() > 1:
    #                 params['RatioStdv'] = ctx_ratio.std()
    #             recorder.tensorboard(global_step, 'Parameters/%s', **params)

    # @property
    # def message(self):
    #     if self._bias_only:
    #         ctx_ratio = self._combine_static.itp_rhs_bias().detach()
    #         if ctx_ratio is not None:
    #             ctx_ratio *= 100
    #             msg = 'Contextual Rate:'
    #             msg += f' {ctx_ratio.mean():.2f}'
    #             if ctx_ratio.nelement() > 1:
    #                 msg += f'±{ctx_ratio.std():.2f}%'
    #             else:
    #                 msg += '%'
    #             return msg