from models.ner import base_config, torch, RnnNer, PadRNN
from models.types import rnn_module_type
from utils.types import hidden_dim, frac_2, frac_4, false_type, num_ori_layer, orient_dim

char_rnn_config = dict(embed_dim    = hidden_dim,
                       drop_out     = frac_4,
                       rnn_drop_out = frac_2,
                       module       = rnn_module_type,
                       num_layers   = num_ori_layer,
                       trainable_initials = false_type)
char_model_type = dict(char_rnn = char_rnn_config)
char_model_type.update(base_config)

class CharNer(RnnNer):
    def __init__(self,
                 num_chars,
                 model_dim,
                 char_rnn,
                 base_layer,
                 ner_extension,
                 num_poses = None,
                 num_bios = None,
                 num_ners = None):
        super().__init__(num_poses,
                         num_bios,
                         num_ners,
                         model_dim,
                         ner_extension,
                         **base_layer)
        self._char_rnn = PadRNN(num_chars, None, None, fence_dim = model_dim, **char_rnn)
    
    def forward(self, sub_idx, sub_fence, pos = None, supervised_fence = None, **kw_args):
        unit_emb = self._char_rnn(sub_idx, sub_fence)
        batch_size, batch_len, model_dim = unit_emb.shape
        seq_len = sub_fence[:, 1:].sum(dim = 1)
        seq_idx = torch.arange(batch_len, device = unit_emb.device)
        existence = seq_idx[None] < seq_len[:, None]
        return (batch_size, batch_len, existence) + super().forward(unit_emb, existence, pos, supervised_fence)