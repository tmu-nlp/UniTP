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
                height = 0,
                **kw_args):
        batch_size, seq_len, _ = existence.shape
        h0c0 = self.get_h0c0(batch_size)

        if height == 0:
            (layers_of_unit, layers_of_existence, layers_of_orient,
             trapezoid_info) = self.triangle_forward(existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)
        else:
            (layers_of_unit, layers_of_existence, layers_of_orient,
             trapezoid_info) = self.trapozoids_forward(height, existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)

        layers_of_unit.reverse()
        layers_of_orient.reverse()
        layers_of_existence.reverse()

        unit_hidden = torch.cat(layers_of_unit,   dim = 1)
        orient      = torch.cat(layers_of_orient, dim = 1)
        existence   = torch.cat(layers_of_existence, dim = 1)

        return StemOutput(unit_hidden, existence, (orient, trapezoid_info))

    def triangle_forward(self,
                         existence,
                         unit_hidden,
                         batch_size, seq_len, h0c0,
                         supervised_orient = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        num_layers = seq_len

        teacher_forcing = isinstance(supervised_orient, Tensor)
        modification = not teacher_forcing and isinstance(supervised_orient, tuple)
        if modification:
            offsets, lengths = supervised_orient
            batch_dim = torch.arange(batch_size, device = existence.device)
            ends = offsets + lengths - 1

        for length in range(num_layers, 0, -1):
            orient = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if length == 1: break

            if teacher_forcing:
                start = s_index(length - 1)
                end   = s_index(length)
                right = supervised_orient[:, start:end, None]
            elif modification: # emprically not necessary, not more such in trapezoids
                right = orient > 0
                starts = torch.where(offsets < length, offsets, torch.zeros_like(offsets))
                _ends_ = ends - (num_layers - length)
                _ends_ = torch.where( starts < _ends_,  _ends_, torch.ones_like(_ends_) * (length - 1))
                right[batch_dim, starts] = True
                right[batch_dim, _ends_] = False
            else:
                right = orient > 0

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)
        return layers_of_unit, layers_of_existence, layers_of_orient, None


    def trapozoids_forward(self,
                           height,
                           existence,
                           unit_hidden,
                           batch_size, seq_len, h0c0,
                           supervised_orient = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        teacher_forcing = isinstance(supervised_orient, Tensor)
        if teacher_forcing:
            end = supervised_orient.shape[1]
        segment, seg_length = [], []

        for l_ in count():
            if not teacher_forcing:
                segment.append(seq_len)
                if l_ % height == 0:
                    seg_length.append(existence.sum(dim = 1)) #
                else:
                    seg_length.append(seg_length[-1] - 1)

            orient = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if seq_len == 1: break

            if teacher_forcing:
                start = end - seq_len
                right = supervised_orient[:, start:end, None]
                end   = start
            else:
                right = orient > 0

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)

            if l_ % height == height - 1:
                # import pdb; pdb.set_trace()
                existence.squeeze_(dim = 2) # will soon be replaced
                helper = condense_helper(existence, as_existence = True)
                unit_hidden, existence = condense_left(unit_hidden, helper, get_cumu = True)
                seq_len = unit_hidden.shape[1]
            else:
                seq_len -= 1

        if not teacher_forcing:
            segment.reverse()
            seg_length.reverse()
            seg_length = torch.cat(seg_length, dim = 1)

        return layers_of_unit, layers_of_existence, layers_of_orient, (segment, seg_length)


from models.loss import get_loss
from models.backend import OutputLayer
from utils.param_ops import change_key
class BaseRnnParser(OutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'orient_layer', 'stem_layer')
        super().__init__(Stem, *args, **kwargs)

    def get_losses(self, batch, tag_logits, top3_label_logits, label_logits, height_mask, weight_mask, key = None):
        tag_fn = self._tag_layer if key is None else self._tag_layer[key]
        label_fn = self._label_layer if key is None else self._label_layer[key]
        tag_loss   = get_loss(tag_fn,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(label_fn, self._logit_max, label_logits, batch, True, height_mask, weight_mask, 'label')
        if top3_label_logits is not None:
            label_loss += get_loss(label_fn, self._logit_max, top3_label_logits, batch, 'top3_label')
        return tag_loss, label_loss