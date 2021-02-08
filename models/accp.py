import torch
from torch import nn, Tensor
from models.backend import Stem, activation_type, logit_type
from models.self_att import SelfAttention

from utils.types import orient_dim, hidden_dim, num_ori_layer, true_type, frac_2, frac_4, frac_5, BaseWrapper, BaseType
from utils.math_ops import inv_sigmoid
from random import random
from sys import stderr


to_name = lambda x: x.__name__
fence_module = BaseType(0, as_index = True, as_exception = True, default_set = BaseWrapper.from_gen((nn.LSTM, nn.GRU), to_name))

from models.combine import get_combinator, get_components, valid_trans_compound
stem_config = dict(fence_dim    = orient_dim,
                   fence_module = fence_module,
                   activation   = activation_type,
                   threshold    = frac_5,
                   num_layers   = num_ori_layer,
                   drop_out     = frac_4,
                   rnn_drop_out = frac_2,
                   trainable_initials = true_type)
from models.loss import cross_entropy, hinge_loss, binary_cross_entropy
from models.utils import hinge_score as hinge_score_
from models.utils import blocky_max, blocky_softmax, birnn_fwbw, fencepost, condense_helper, condense_left

class MAryStem(nn.Module):
    def __init__(self,
                 model_dim,
                 fence_dim,
                 fence_module,
                 activation,
                 threshold,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 trainable_initials):
        super().__init__()
        hidden_size = fence_dim // 2
        self._fence_emb = fence_module(model_dim, hidden_size,
                                       num_layers    = num_layers,
                                       bidirectional = True,
                                       batch_first   = True,
                                       dropout = rnn_drop_out if num_layers > 1 else 0)
        if trainable_initials:
            c0 = torch.randn(num_layers * 2, 1, hidden_size)
            h0 = torch.randn(num_layers * 2, 1, hidden_size)
            self._c0 = nn.Parameter(c0, requires_grad = True)
            self._h0 = nn.Parameter(h0, requires_grad = True)
            self._h0_act = nn.Tanh()
            self._initial_size = hidden_size
        else:
            self.register_parameter('_h0', None)
            self.register_parameter('_c0', None)
            self._initial_size = None

        pad = torch.empty(1, 1, hidden_size)
        self._pad = nn.Parameter(pad, requires_grad = True)
        self._tanh = nn.Tanh()
        self._raw_threshold = threshold - 0.5 # TODO this should be raw
        self._stem_dp = nn.Dropout(drop_out)
        self._fence_l1 = nn.Linear(fence_dim, hidden_size)
        self._fence_act = activation()
        self._fence_l2 = nn.Linear(hidden_size, 1)
        self._domain = nn.Linear(fence_dim, model_dim)
        self._subject_fw = nn.Linear(hidden_size, model_dim)
        self._subject_bw = nn.Linear(hidden_size, model_dim, bias = False)
        self._subject_fwd = nn.Linear(hidden_size, model_dim, bias = False)
        self._subject_bwd = nn.Linear(hidden_size, model_dim, bias = False)
        # self._subject_bfw = nn.Linear(hidden_size, model_dim, bias = False)
        # self._subject_bbw = nn.Linear(hidden_size, model_dim, bias = False)
        # self._subject = nn.Linear(fence_dim, model_dim)
        # self._subject_static = nn.Linear(model_dim, model_dim, bias = False)
        self._sigmoid = nn.Sigmoid()
        finfo = torch.finfo(torch.get_default_dtype())
        self._fminmax = finfo.min, finfo.max

    @property
    def threshold(self):
        return self._raw_threshold

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0_act(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def predict_fence(self, unit_hidden, h0c0, seq_len):
        fence_hidden, _ = self._fence_emb(unit_hidden, h0c0)
        fence_hidden = self._stem_dp(fence_hidden)
        fw, bw = birnn_fwbw(fence_hidden, seq_len = seq_len, pad = self._tanh(self._pad))
        fence = torch.cat([fw, bw], dim = 2)
        fence = self._fence_l1(fence)
        fence = self._stem_dp(fence)
        fence = self._fence_act(fence)
        fence = self._fence_l2(fence).squeeze(dim = 2)
        return fence, fence_hidden, fw, bw
    
    def forward(self, unit_emb, existence,
                supervised_fence = None,
                keep_low_attention_rate = 1,
                **kw_args):
        batch_size, seg_len = existence.shape
        h0c0 = self.get_h0c0(batch_size)
        max_iter_n = seg_len << 2 # 4 times
        teacher_forcing = isinstance(supervised_fence, list)
        segment, seg_length = [], []
        batch_dim = torch.arange(batch_size, device = unit_emb.device)
        
        layers_of_u_emb = []
        layers_of_fence = []
        layers_of_existence = []
        layers_of_weight = []
        layers_of_fence_idx = []

        for l_cnt in range(max_iter_n):
            seq_len = existence.sum(dim = 1)
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(seq_len)

            fence_logits, fence_hidden, fw, bw = self.predict_fence(unit_emb, h0c0, seq_len)
            layers_of_u_emb.append(unit_emb)
            layers_of_existence.append(existence)
            if seg_len == 1:
                break # teacher forcing or a good model
            elif len(seg_length) > 1:
                prev, curr = seg_length[-2:]
                if (prev == curr).all():
                    break
                elif l_cnt == max_iter_n - 1:
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break

            longer_seq_idx = torch.arange(seg_len + 1, device = unit_emb.device)[None, :]
            if teacher_forcing:
                fence_idx = supervised_fence[l_cnt]

                sections = torch.zeros(batch_size, seg_len + 1, dtype = torch.bool, device = unit_emb.device)
                sections[batch_dim[:, None], fence_idx] = True
                sections = sections.cumsum(dim = 1)
            else:
                fmin, fmax = self._fminmax
                fence_logits[:, 0] = fmax
                fence_logits[batch_dim, seq_len] = fmax
                fence_logits[longer_seq_idx > seq_len[:, None]] = fmin
                fence = fence_logits > self._raw_threshold
                idx = longer_seq_idx * fence
                helper = condense_helper(fence, as_existence = True)
                fence_idx = condense_left(idx, helper)
                layers_of_fence_idx.append(fence_idx)

                sections = fence.cumsum(dim = 1)
            
            dom_emb = self._domain(fencepost(fw, bw, fence_idx))
            dom_emb = self._stem_dp(dom_emb)
            # sub_emb  = self._subject_ffw(fw[:, 1:]) + self._subject_fbw(fw[:, :-1]) + self._subject_bfw(bw[:, :-1]) + self._subject_bbw(bw[:, 1:])
            #* self._sigmoid(self._subject_static(unit_emb)) #* 20
            sub_emb = self._stem_dp(self._subject_fwd(fw[:, 1:] - fw[:, :-1]))
            sub_emb = sub_emb + self._stem_dp(self._subject_bwd(bw[:, :-1] - bw[:, 1:]))
            sub_emb = sub_emb + self._stem_dp(self._subject_fw(fw[:, :-1]))
            sub_emb = sub_emb + self._stem_dp(self._subject_bw(bw[:, 1:]))
            # sub_emb = self._stem_dp(sub_emb)
            sections = torch.where(longer_seq_idx < seq_len[:, None], sections, torch.zeros_like(sections))[:, :-1]

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
            weight = None
        else:
            weight     = torch.cat(layers_of_weight,    dim = 1)
            fence_idx  = torch.cat(layers_of_fence_idx, dim = 1)
            seg_length = torch.stack(seg_length, dim = 1)

        return existence, embeddings, weight, fence, fence_idx, segment, seg_length
    

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(fence_layer    = stem_config,
                  tag_label_layer = multi_class)
from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value, get_loss

class BaseRnnTree(MAryStem):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 fence_layer,
                 tag_label_layer,
                 **kw_args):
        # (**kw_args)self._stem_layer = 

        super().__init__(model_dim, **fence_layer)

        hidden_dim = tag_label_layer['hidden_dim']
        if hidden_dim:
            self._shared_layer = nn.Linear(model_dim, hidden_dim)
            self._dp_layer = nn.Dropout(tag_label_layer['drop_out'])

            Net, argmax, score_act = get_logit_layer(tag_label_layer['logit_type'])
            self._tag_layer   = Net(hidden_dim, num_tags) if num_tags else None
            self._label_layer = Net(hidden_dim, num_labels) if num_labels else None
            self._logit_max = argmax
            if argmax:
                self._activation = tag_label_layer['activation']()
            self._score_fn = score_act(dim = 2)
        self._hidden_dim = hidden_dim
        self._model_dim = model_dim

    def forward(self,
                base_inputs,
                bottom_existence,
                ingore_logits = False,
                **kw_args):
        (existence, embeddings, weight, fence, fence_idx, segment,
         seg_length) = super().forward(base_inputs, bottom_existence, **kw_args)

        if self._hidden_dim:
            layers_of_hidden = self._shared_layer(embeddings)
            layers_of_hidden = self._dp_layer(layers_of_hidden)
            if self._logit_max:
                layers_of_hidden = self._activation(layers_of_hidden)

            if self._tag_layer is None or ingore_logits:
                tags = None
            else:
                _, batch_len, _ = base_inputs.shape
                tags = self._tag_layer(layers_of_hidden[:, :batch_len]) # diff small endian
            
            if self._label_layer is None or ingore_logits:
                labels = None
            else:
                labels = self._label_layer(layers_of_hidden)
        else:
            layers_of_hidden = tags = labels = None

        return existence, embeddings, weight, fence, fence_idx, tags, labels, segment, seg_length

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_label(self, hidden):
        return self._label_layer(hidden)

    def get_decision(self, logits):
        return get_decision(self._logit_max, logits)

    def get_decision_with_value(self, logits):
        return get_decision_with_value(self._score_fn, logits)

    def get_losses(self, batch, weight_mask, tag_logits, label_logits):
        height_mask = batch['segment'][None] * (batch['seg_length'] > 0)
        height_mask = height_mask.sum(dim = 1)
        tag_loss   = get_loss(self._tag_layer,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(self._label_layer, self._logit_max, label_logits, batch, False, height_mask, weight_mask, 'label')
        # height mask and weight_mask are both beneficial!
        return tag_loss, label_loss