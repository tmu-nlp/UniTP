import torch
from torch import nn, Tensor
from models.combine import get_combinator, combine_type
from models.types import activation_type, logit_type
from utils.types import hidden_dim, orient_dim, num_ori_layer, frac_2, frac_4, false_type

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   num_layers   = num_ori_layer,
                   rnn_drop_out = frac_2,
                   drop_out     = frac_4,
                   trainable_initials = false_type)

model_type = dict(orient_layer    = stem_config,
                  tag_label_layer = multi_class)
from utils.math_ops import s_index
from itertools import count
from models.utils import condense_helper, condense_left
from models import StemOutput
class Stem(nn.Module):
    def __init__(self,
                 model_dim,
                 orient_dim,
                 combine_type,
                 num_layers,
                 rnn_drop_out,
                 trainable_initials,
                 drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        self.orient_emb = nn.LSTM(model_dim, hidden_size,
                                  num_layers    = num_layers,
                                  bidirectional = True,
                                  batch_first   = True,
                                  dropout = rnn_drop_out if num_layers > 1 else 0)
        self._dp_layer = nn.Dropout(drop_out)
        self.orient = nn.Linear(orient_dim, 1)
        self.combine = get_combinator(combine_type, model_dim)
        self._threshold = 0
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

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def set_threshold(self, threshold):
        self._threshold = threshold

    def orientation(self, logits):
        return logits > self._threshold

    def blind_combine(self, unit_hidden, existence = None):
        return self.combine(None, unit_hidden, existence)

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0_act(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def predict_orient(self, unit_hidden, h0c0):
        orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
        orient_hidden = self._dp_layer(orient_hidden)
        return self.orient(orient_hidden)

    def forward(self,
                existence,
                unit_hidden,
                condense_per,
                **kw_args):
        batch_size, seq_len, _ = existence.shape
        h0c0 = self.get_h0c0(batch_size)

        if not condense_per:
            (layers_of_unit, layers_of_existence, layers_of_orient, batch_segment,
             segment) = self.triangle_forward(existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)
        else:
            (layers_of_unit, layers_of_existence, layers_of_orient, batch_segment,
             segment) = self.trapozoids_forward(condense_per, existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)

        unit_hidden = torch.cat(layers_of_unit,   dim = 1)
        orient      = torch.cat(layers_of_orient, dim = 1)
        existence   = torch.cat(layers_of_existence, dim = 1)

        return StemOutput(unit_hidden, existence, batch_segment, (orient, segment))

    def triangle_forward(self,
                         existence,
                         unit_hidden,
                         batch_size, seq_len, h0c0,
                         supervision = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        num_layers = seq_len

        teacher_forcing = isinstance(supervision, Tensor)
        modification = not teacher_forcing and isinstance(supervision, tuple)
        if modification:
            offsets, lengths = supervision
            if offsets is None: offsets = torch.zeros_like(lengths)
            seq_dim = torch.arange(seq_len, device = existence.device)[None]
            seq_end = offsets + lengths

        start = 0
        segment = []
        for reduce_num in range(num_layers):
            length = num_layers - reduce_num
            orient_logits = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient_logits)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            segment.append(length)
            if length == 1: break

            if teacher_forcing:
                end   = start + length
                right = supervision[:, start:end, None]
                start = end
            elif modification: # emprically not necessary, not more such in trapezoids
                seq = seq_dim[:length]
                bos = seq <= offsets
                eos = seq >= (seq_end - reduce_num - 1)
                right[eos] = False
                right[bos] = True
            else:
                right = self.orientation(orient_logits)

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)
        return layers_of_unit, layers_of_existence, layers_of_orient, segment, None


    def trapozoids_forward(self,
                           condense_per,
                           existence,
                           unit_hidden,
                           _, seq_len, h0c0,
                           supervision = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        teacher_forcing = isinstance(supervision, Tensor)
        start = 0
        batch_segment, segment = [], []

        for lid in count():
            batch_segment.append(seq_len)
            if not teacher_forcing or lid % condense_per == 0:
                segment.append(existence.sum(dim = 1)) #
            else:
                segment.append(segment[-1] - 1)

            orient_logits = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient_logits)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if seq_len == 1: break

            if teacher_forcing:
                end   = start + seq_len
                right = supervision[:, start:end, None]
                start = end
            else:
                right = self.orientation(orient_logits)

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)

            if not teacher_forcing or lid % condense_per == condense_per - 1:
                existence.squeeze_(dim = 2)
                helper = condense_helper(existence, as_existence = True)
                unit_hidden, existence = condense_left(unit_hidden, helper, get_cumu = True)
                seq_len = unit_hidden.shape[1]
            else:
                seq_len -= 1

        return layers_of_unit, layers_of_existence, layers_of_orient, batch_segment, torch.cat(segment, dim = 1)


from models.backend import OutputLayer
from utils.param_ops import change_key
class _CB(OutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'orient_layer', 'stem_layer')
        super().__init__(Stem, *args, **kwargs)