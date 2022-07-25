import torch
from torch import nn

from utils.types import chunk_dim, hidden_dim, half_hidden_dim, num_ori_layer, frac_2, frac_4, false_type
from sys import stderr

from models.types import rnn_module_type, continuous_attention_hint, activation_type, logit_type, fmin, fmax, fence_vote
stem_config = dict(fence_dim      = chunk_dim,
                   fence_module   = rnn_module_type,
                   fence_vote     = fence_vote,
                   linear_dim     = half_hidden_dim,
                   activation     = activation_type,
                   attention_hint = continuous_attention_hint,
                   num_layers     = num_ori_layer,
                   drop_out       = frac_4,
                   rnn_drop_out   = frac_2,
                   trainable_initials = false_type)
from models.utils import blocky_max, blocky_softmax, condense_helper, condense_left
from models.backend import PadRNN
from models import StemOutput

class MultiStem(PadRNN):
    def __init__(self,
                 model_dim,
                 fence_dim,
                 linear_dim,
                 fence_module,
                 fence_vote,
                 activation,
                 attention_hint,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 trainable_initials):
        super().__init__(None,
                         attention_hint,
                         linear_dim,
                         model_dim,
                         fence_dim,
                         drop_out,
                         num_layers,
                         fence_module,
                         rnn_drop_out,
                         trainable_initials,
                         fence_vote,
                         activation)
        self._sigmoid = nn.Sigmoid()

    def forward(self, existence, unit_emb,
                supervised_fence = None,
                keep_low_attention_rate = 1,
                **kw_args):
        batch_size, seg_len = existence.shape
        h0c0 = self.get_h0c0(batch_size)
        max_iter_n = seg_len << 2 # 4 times
        teacher_forcing = isinstance(supervised_fence, list)
        segment, seg_length = [], []
        batch_dim = torch.arange(batch_size, device = unit_emb.device)

        if self._fence_vote is None:
            layers_of_vote = None
        else:
            layers_of_vote = []
        
        layers_of_u_emb = []
        layers_of_fence = []
        layers_of_existence = []
        layers_of_weight = []
        layers_of_fence_idx = []

        for l_cnt in range(max_iter_n):
            seq_len = existence.sum(dim = 1)
            layers_of_u_emb.append(unit_emb)
            layers_of_existence.append(existence)
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(seq_len)

            if seg_len == 1:
                break # teacher forcing or a good model
            elif len(seg_length) > 1:
                prev, curr = seg_length[-2:]
                if (prev == curr).all():
                    break
                elif l_cnt == max_iter_n - 1:
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break

            fence_hidden, _ = self._fence_emb(unit_emb, h0c0)
            fw, bw = self.pad_fwbw_hidden(fence_hidden, existence)
            fence_hidden = self._stem_dp(fence_hidden)
            fw = self._stem_dp(fw)
            bw = self._stem_dp(bw)
            if self._fence_vote is None:
                fence_logits = self.predict_fence(fw, bw)
            else:
                from_unit, method = self._fence_vote
                votes, fence_logits = method(fw, bw, unit_emb if from_unit else fence_hidden, seq_len)
                layers_of_vote.append(votes.reshape(batch_size, -1)) # [b, s+1, s]
            longer_seq_idx = torch.arange(seg_len + 1, device = unit_emb.device)[None, :]
            
            if teacher_forcing:
                fence_idx = supervised_fence[l_cnt]

                sections = torch.zeros(batch_size, seg_len + 1, dtype = torch.bool, device = unit_emb.device)
                sections[batch_dim[:, None], fence_idx] = True
                sections = sections.cumsum(dim = 1)
            else:
                fence_logits[:, 0] = fmax
                fence_logits[batch_dim, seq_len] = fmax
                fence_logits[longer_seq_idx > seq_len[:, None]] = fmin
                fence = fence_logits > 0
                idx = longer_seq_idx * fence
                helper = condense_helper(fence, as_existence = True)
                fence_idx = condense_left(idx, helper)
                layers_of_fence_idx.append(fence_idx)

                sections = fence.cumsum(dim = 1)
            dom_emb, sub_emb = self.domain_and_subject(fw, bw, fence_idx, unit_emb, fence_hidden)
            sections = torch.where(longer_seq_idx < seq_len[:, None], sections, torch.zeros_like(sections))[:, :-1]

            #* self._sigmoid(self._subject_static(unit_emb)) #* 20
            if keep_low_attention_rate < 1:
                max_mask = blocky_max(sections, sub_emb.mean(dim = 2))
                max_mask |= torch.rand(batch_size, seg_len, device = sub_emb.device) < keep_low_attention_rate
                max_mask |= torch.rand(batch_size, seg_len, device = sub_emb.device) < self._sigmoid(sub_emb.sum(dim = 2))
                sub_emb = torch.where(max_mask[:, :, None], sub_emb, sub_emb - (sub_emb.max() - sub_emb.min()) * 0.7) # max must be kept
                
            weights, unit_emb = blocky_softmax(sections, sub_emb, dom_emb, unit_emb)
            seg_len  = unit_emb.shape[1]
            existence = fence_idx[:, 1:] > 0
            layers_of_weight.append(weights)
            layers_of_fence.append(fence_logits)

        embeddings = torch.cat(layers_of_u_emb, dim = 1)
        fence      = torch.cat(layers_of_fence, dim = 1)
        existence  = torch.cat(layers_of_existence, dim = 1)
        if teacher_forcing:
            weight     = None
            fence_vote = None
        else:
            weight     = torch.cat(layers_of_weight,    dim = 1)
            fence_idx  = torch.cat(layers_of_fence_idx, dim = 1)
            seg_length = torch.stack(seg_length, dim = 1)
            if self._fence_vote is None:
                fence_vote = None
            elif layers_of_vote:
                fence_vote = torch.cat(layers_of_vote, dim = 1)
            else:
                fence_vote = torch.zeros(batch_size, 0, dtype = unit_emb.shape, device = unit_emb.device)

        return StemOutput(embeddings, existence, (weight, fence, fence_idx, fence_vote, segment, seg_length))

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(fence_layer     = stem_config,
                  tag_label_layer = multi_class)

from models.loss import get_loss
from models.backend import OutputLayer
from utils.param_ops import change_key
class BaseRnnParser(OutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'fence_layer', 'stem_layer')
        super().__init__(MultiStem, *args, **kwargs)

    def get_losses(self, batch, weight_mask, tag_logits, label_logits, key = None):
        tag_fn = self._tag_layer if key is None else self._tag_layer[key]
        label_fn = self._label_layer if key is None else self._label_layer[key]
        height_mask = batch['segment'][None] * (batch['seg_length'] > 0)
        height_mask = height_mask.sum(dim = 1)
        tag_loss   = get_loss(tag_fn,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(label_fn, self._logit_max, label_logits, batch, False, height_mask, weight_mask, 'label')
        # height mask and weight_mask are both beneficial! (nop, weight_mask by freq is not helping)
        return tag_loss, label_loss