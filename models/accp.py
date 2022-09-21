import torch
from torch import nn

from utils.types import chunk_dim, hidden_dim, half_hidden_dim, num_ori_layer, frac_2, frac_4, false_type
from sys import stderr

from models.types import rnn_module_type, continuous_attention_hint, activation_type, logit_type, fmin, fmax, chunk_vote
stem_config = dict(chunk_dim      = chunk_dim,
                   chunk_module   = rnn_module_type,
                   chunk_vote     = chunk_vote,
                   linear_dim     = half_hidden_dim,
                   activation     = activation_type,
                   attention_hint = continuous_attention_hint,
                   num_layers     = num_ori_layer,
                   drop_out       = frac_4,
                   rnn_drop_out   = frac_2,
                   trainable_initials = false_type)
from models.utils import blocky_max, blocky_softmax, condense_helper, condense_left
from models.backend import PadRNN, simple_parameters
from models import StemOutput

class MultiStem(PadRNN):
    def __init__(self,
                 model_dim,
                 chunk_dim,
                 linear_dim,
                 chunk_module,
                 chunk_vote,
                 char_chunk,
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
                         chunk_dim,
                         drop_out,
                         num_layers,
                         chunk_module,
                         rnn_drop_out,
                         trainable_initials,
                         chunk_vote,
                         activation)
        self._threshold = 0
        self._sigmoid = nn.Sigmoid()
        self._char_bias = simple_parameters(1, 1, model_dim) if char_chunk else None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def set_threshold(self, threshold):
        self._threshold = threshold

    def chunk(self, logits):
        return logits > self._threshold

    def forward(self, existence, embedding,
                n_layers = 0,
                supervision = None,
                bottom_supervision = None,
                keep_low_attention_rate = 1,
                **kw_args):
        batch_segment, segment = [], []
        batch_size, layer_len = existence.shape
        h0c0 = self.get_h0c0(batch_size)
        max_iteration = layer_len + (layer_len >> 1) # 1.5 times
        chunk_seq = torch.arange(layer_len + 1, device = embedding.device)
        if not (teacher_forcing := isinstance(supervision, torch.Tensor)):
            batch_dim = torch.arange(batch_size, device = embedding.device)
            if bottom_supervision:
                n_layers, supervision = bottom_supervision

        if self._chunk_vote is None:
            layers_of_vote = None
        else:
            layers_of_vote = []

        start = 0
        layers_of_chunk = []
        layers_of_weight = []
        layers_of_existence = []
        layers_of_embedding = []

        for l_cnt in range(max_iteration):
            length = existence.sum(dim = 1)
            layers_of_embedding.append(embedding)
            layers_of_existence.append(existence)
            batch_segment.append(layer_len)
            segment.append(length)

            if layer_len == 1:
                break
            elif len(segment) > 1:
                prev, curr = segment[-2:]
                if (prev == curr).all():
                    break
                elif l_cnt == max_iteration - 1:
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break

            if l_cnt < n_layers:
                chunk_hidden, _ = self._chunk_emb(embedding + self._char_bias, h0c0)
            else:
                chunk_hidden, _ = self._chunk_emb(embedding, h0c0)
            fw_hidden, bw_hidden = self.pad_fwbw_hidden(chunk_hidden, existence)
            fw_hidden, bw_hidden = self._stem_dp(fw_hidden), self._stem_dp(bw_hidden)
            chunk_hidden = self._stem_dp(chunk_hidden)
            chunk_dim = chunk_seq[None, :layer_len + 1]
            if self._chunk_vote is None:
                chunk_logits = self.predict_chunk(fw_hidden, bw_hidden)
            else:
                from_unit, method = self._chunk_vote
                votes, chunk_logits = method(fw_hidden, bw_hidden, embedding if from_unit else chunk_hidden, length)
                layers_of_vote.append(votes.reshape(batch_size, -1)) # [b, s+1, s]
            
            if teacher_forcing or bottom_supervision and l_cnt < n_layers:
                end = start + layer_len + 1
                chunk = supervision[:, start:end]
                start = end
                if not teacher_forcing:
                    chunk_logits = chunk.type(chunk_logits.dtype) - 0.5
                elif kw_args['batch_segment'][l_cnt] != layer_len:
                    breakpoint()
            else:
                chunk_logits[:, 0] = fmax
                chunk_logits[batch_dim, length] = fmax
                chunk_logits[chunk_dim > length[:, None]] = fmin
                chunk = self.chunk(chunk_logits)

            space = chunk.cumsum(dim = 1)
            split_fn = lambda: condense_left(chunk_dim * chunk, condense_helper(chunk, as_existence = True))
            dom_emb, sub_emb = self.domain_and_subject(fw_hidden, bw_hidden, split_fn, embedding, chunk_hidden)
            space = torch.where(chunk_dim < length[:, None], space, torch.zeros_like(space))[:, :-1]

            #* self._sigmoid(self._subject_static(embedding)) #* 20
            if keep_low_attention_rate < 1:
                max_mask = blocky_max(space, sub_emb.mean(dim = 2))
                max_mask |= torch.rand(batch_size, layer_len, device = sub_emb.device) < keep_low_attention_rate
                max_mask |= torch.rand(batch_size, layer_len, device = sub_emb.device) < self._sigmoid(sub_emb.sum(dim = 2))
                sub_emb = torch.where(max_mask[:, :, None], sub_emb, sub_emb - (sub_emb.max() - sub_emb.min()) * 0.7) # max must be kept
                
            weights, embedding = blocky_softmax(space, sub_emb, dom_emb, embedding)
            layer_len = embedding.shape[1]
            existence = torch.arange(layer_len, device = sub_emb.device)[None] < chunk[:, 1:].sum(dim = 1, keepdim = True)
            layers_of_weight .append(weights)
            layers_of_chunk  .append(chunk_logits)

        chunk     = torch.cat(layers_of_chunk, dim = 1)
        segment   = torch.stack(segment, dim = 1)
        embedding = torch.cat(layers_of_embedding, dim = 1)
        existence = torch.cat(layers_of_existence, dim = 1)
        if teacher_forcing:
            weight     = None
            chunk_vote = None
        else:
            weight = torch.cat(layers_of_weight, dim = 1)
            if self._chunk_vote is None:
                chunk_vote = None
            elif layers_of_vote:
                chunk_vote = torch.cat(layers_of_vote, dim = 1)
            else:
                chunk_vote = torch.zeros(batch_size, 0, dtype = embedding.shape, device = embedding.device)

        return StemOutput(embedding, existence, batch_segment, (weight, chunk, chunk_vote, segment))

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(chunk_layer     = stem_config,
                  tag_label_layer = multi_class)

from models.loss import get_loss
from models.backend import ParsingOutputLayer
from utils.param_ops import change_key
class _CM(ParsingOutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'chunk_layer', 'stem_layer')
        super().__init__(MultiStem, *args, **kwargs)