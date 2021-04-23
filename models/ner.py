from models.backend import PadRNN, torch, nn
from models.types import rnn_module_type, fmin, fmax, continuous_attention_hint, activation_type, fence_vote
from models.utils import blocky_softmax
from utils.types import true_type, false_type, hidden_dim, num_ori_layer, frac_2, frac_4, orient_dim
from utils.param_ops import HParams

bio_model_config = dict(hidden_dim = hidden_dim,
                        num_layers = num_ori_layer,
                        drop_out = frac_4,
                        module = rnn_module_type,
                        rnn_drop_out = frac_2,
                        trainable_initials = false_type,
                        pos_as_input = true_type,
                        tag_dim = hidden_dim,
                        tag_activation = activation_type)
# num_ners, model_dim is offered by t_xxxx_ner/__init__.py

ner_model_config = dict(attention_hint = continuous_attention_hint,
                        fence_activation = activation_type,
                        fence_dim  = hidden_dim, # TODO open for 1
                        fence_vote = fence_vote)

base_config = dict(base_layer = bio_model_config,
                   ner_extension = ner_model_config,
                   model_dim = hidden_dim)

class RnnNer(PadRNN):
    def __init__(self,
                 num_poses,
                 num_bios,
                 num_ners,
                 model_dim,
                 ner_extension,
                 pos_as_input,
                 hidden_dim,
                 tag_dim,
                 tag_activation,
                 num_layers,
                 drop_out,
                 module = None,
                 rnn_drop_out = None,
                 trainable_initials = False):
        if not num_layers:
            assert model_dim == hidden_dim
        ner_extension = HParams(ner_extension, True)
        super().__init__(None,
                         ner_extension.attention_hint,
                         None if num_bios else ner_extension.fence_dim,
                         model_dim,
                         hidden_dim,
                         drop_out,
                         num_layers,
                         module,
                         rnn_drop_out,
                         trainable_initials,
                         ner_extension.fence_vote,
                         ner_extension.fence_activation)
        self._tag_act = act = tag_activation()
        if num_poses:
            if pos_as_input:
                self._pos = nn.Embedding(num_poses, model_dim)
            elif tag_dim == 1:
                self._pos = nn.Linear(hidden_dim, num_poses)
            else:
                self._pos_l1 = pl1 = nn.Linear(hidden_dim, tag_dim)
                self._pos_l2 = pl2 = nn.Linear(tag_dim, num_poses)
                self._pos = lambda x: pl2(act(self._stem_dp(pl1(x))))
        self._pos_configs = num_poses, num_poses and pos_as_input
        if num_bios:
            assert not num_ners
            if tag_dim == 1:
                self._bio = nn.Linear(hidden_dim, num_bios)
            else:
                self._bio_l1 = bl1 = nn.Linear(hidden_dim, tag_dim)
                self._bio_l2 = bl2 = nn.Linear(tag_dim, num_bios)
                self._bio = lambda x: bl2(act(self._stem_dp(bl1(x))))
        else:
            assert not num_bios
            self._bio = None
            if tag_dim == 1:
                self._ner = nn.Linear(model_dim, num_ners)
            else:
                self._ner_l1 = nl1 = nn.Linear(model_dim, tag_dim)
                self._ner_l2 = nl2 = nn.Linear(tag_dim, num_ners)
                self._ner = lambda x: nl2(act(self._stem_dp(nl1(x))))

    def forward(self, unit_emb, existence = None, pos = None, supervised_fence = None, **kw_args):
        has_pos, pos_as_input = self._pos_configs
        if pos_as_input:
            unit_emb = unit_emb + self._stem_dp(self._pos(pos))
        batch_size, batch_len, model_dim = unit_emb.shape
        if self._fence_emb is None:
            state = unit_emb
        else:
            state, _ = self._fence_emb(unit_emb, self.get_h0c0(batch_size))
        if has_pos and not pos_as_input:
            pos_logits = self._pos(self._stem_dp(state))
        else:
            pos_logits = None
        if self._bio:
            bio_logits = self._bio(self._stem_dp(state))
            ner_logits = fence_logits = weights = fences = None
        else:
            fw, bw = self.pad_fwbw_hidden(self._stem_dp(state), existence)
            fw = self._stem_dp(fw)
            bw = self._stem_dp(bw)
            fence_logits = self.predict_fence(fw, bw)
            if supervised_fence is None:
                batch_dim = torch.arange(batch_size, device = unit_emb.device)
                seq_len = existence.sum(dim = 1)
                fence_logits[batch_dim, 0] = fmax
                fence_logits[batch_dim, seq_len] = fmax
                fences = fence_logits > 0
                fence = fences[:, :batch_len]
            else:
                fences = fence_logits > 0
                fence = supervised_fence[:, :batch_len]

            sections = fence.cumsum(dim = 1) * existence
            dom_emb, sub_emb = self.domain_and_subject(fw, bw, None, unit_emb, state)
            weights, unit_emb = blocky_softmax(sections, sub_emb, dom_emb, unit_emb)
            ner_logits = self._ner(unit_emb)
            bio_logits = None
        return pos_logits, bio_logits, ner_logits, fence_logits, fences, weights