import torch
from torch import nn, Tensor
from utils.math_ops import s_index

from utils.types import BaseType, true_type, frac_4
from utils.types import orient_dim, combine_type

contextual_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.LSTM, nn.GRU))
activation_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.ReLU, nn.Sigmoid, nn.Tanh))

stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   drop_out     = frac_4,
                   trainable_initials = true_type)

from models.combine import get_combinator
from models.utils import squeeze_left
from itertools import count
class Stem(nn.Module):
    def __init__(self,
                 word_dim,
                 orient_dim,
                 combine_type,
                 trainable_initials,
                 drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        self.orient_emb = nn.LSTM(word_dim, hidden_size,
                                  bidirectional = True,
                                  batch_first   = True)
        self._dp_layer = nn.Dropout(drop_out)
        self.orient = nn.Linear(orient_dim, 1)
        self.combine = get_combinator(combine_type, word_dim)
        if trainable_initials:
            c0 = torch.randn(1 * 2, 1, hidden_size)
            h0 = torch.randn(1 * 2, 1, hidden_size)
            self._c0 = nn.Parameter(c0, requires_grad = True)
            self._h0 = nn.Parameter(h0, requires_grad = True)
            self._initial_size = hidden_size
        else:
            self.register_parameter('_h0', None)
            self.register_parameter('_c0', None)
            self._initial_size = None

    def forward(self,
                existence,
                unit_hidden,
                height = 0,
                **kw_args):
        batch_size, seq_len = existence.shape
        existence.unsqueeze_(dim = -1)
        unit_hidden *= existence
        
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0c0 = h0, c0
        else:
            h0c0 = None

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

        return unit_hidden, orient, existence, trapezoid_info

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
            orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
            orient_hidden = self._dp_layer(orient_hidden)
            orient = self.orient(orient_hidden)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if length == 1: break

            if teacher_forcing:
                start = s_index(length - 1)
                end   = s_index(length)
                right = supervised_orient[:, start:end, None]
            elif modification:
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
        return (layers_of_unit, layers_of_existence, layers_of_orient, None)


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
                
            orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
            orient_hidden = self._dp_layer(orient_hidden)
            orient = self.orient(orient_hidden)
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
                unit_hidden, existence = squeeze_left(unit_hidden, existence, as_existence = True)
                seq_len = unit_hidden.shape[1]
            else:
                seq_len -= 1

        if not teacher_forcing:
            segment.reverse()
            seg_length.reverse()
            seg_length = torch.cat(seg_length, dim = 1)
            
        return (layers_of_unit, layers_of_existence, layers_of_orient, (segment, seg_length))