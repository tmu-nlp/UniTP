import torch
from torch import nn, Tensor
from utils.math_ops import s_index

from utils.types import BaseType, true_type, frac_4, frac_2
from utils.types import orient_dim, combine_type, num_ori_layer

contextual_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.LSTM, nn.GRU))
activation_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.ReLU, nn.ReLU6, nn.Softplus,# end == 0, nn.GELU
                                                                                   nn.LeakyReLU, nn.ELU, nn.CELU, nn.SELU, nn.RReLU, # end < 0
                                                                                   nn.Sigmoid, nn.LogSigmoid,
                                                                                   nn.Tanh, nn.Softsign, nn.Hardtanh, # -<0<+
                                                                                   nn.Tanhshrink, nn.Softshrink, nn.Hardshrink)) # -0+

stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   num_layers   = num_ori_layer,
                   rnn_drop_out = frac_2,
                   drop_out     = frac_4,
                   trainable_initials = true_type)

from models.combine import get_combinator
from models.utils import squeeze_left
from itertools import count
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

    @property
    def combinator(self):
        return self.combine
    
    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0_act(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def forward_layer(self, unit_hidden, h0c0):
        orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
        orient_hidden = self._dp_layer(orient_hidden)
        return self.orient(orient_hidden)

    def forward(self,
                existence,
                unit_hidden,
                height = 0,
                **kw_args):
        batch_size, seq_len = existence.shape
        existence.unsqueeze_(dim = -1)
        unit_hidden *= existence
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
            orient = self.forward_layer(unit_hidden, h0c0)
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
                
            orient = self.forward_layer(unit_hidden, h0c0)
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


from models.utils import PCA
from utils.types import false_type, num_ctx_layer
from utils.param_ops import HParams, dict_print
act_fasttext = BaseType(None, as_index = True, as_exception = True, default_set = (nn.Tanh, nn.Softsign))
input_config = dict(pre_trained = true_type, activation = act_fasttext, trainable = false_type, drop_out = frac_4)

class InputLeaves(nn.Module):
    def __init__(self,
                 model_dim,
                 num_types,
                 initial_weight,
                 pre_trained,
                 trainable,
                 activation,
                 drop_out):
        super().__init__()

        st_dy_bound = 0
        st_emb_layer = dy_emb_layer = None
        if pre_trained:
            num_special_tokens = num_types - initial_weight.shape[0]
            assert num_special_tokens >= 0
            if num_special_tokens > 0: # bos eos
                if trainable:
                    initial_weight = torch.cat([torch.tensor(initial_weight), torch.rand(num_special_tokens, model_dim)], 0)
                    st_emb_layer = None
                    dy_emb_layer = nn.Embedding.from_pretrained(initial_weight)
                else:
                    assert model_dim == initial_weight.shape[1]
                    st_dy_bound = initial_weight.shape[0]
                    st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
                    dy_emb_layer = nn.Embedding(num_special_tokens, model_dim)
            elif trainable: # nil nil
                dy_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
            else: # nil nil
                st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
        else: # nil ... unk | ... unk bos eos
            assert trainable
            dy_emb_layer = nn.Embedding(num_types, model_dim)

        if activation is None:
            self._act_pre_trained = None
        else:
            self._act_pre_trained = activation()
        self._dp_layer = nn.Dropout(drop_out)
        self._model_dim = model_dim
        self._st_dy_bound = st_dy_bound
        self._st_emb_layer = st_emb_layer
        self._dy_emb_layer = dy_emb_layer
        self._pca_base = None

    def pca(self, word_emb, flush = False):
        # TODO: setup_pca with external
        if flush or self._st_emb_layer is not None and self._pca_base is None:
            self._pca_base = PCA(self._st_emb_layer.weight)
        return self._pca_base(word_emb)

    def forward(self, word_idx):
        if self._st_dy_bound > 0:
            b_ = self._st_dy_bound
            c_ = word_idx < b_
            st_idx = torch.where(c_, word_idx, torch.zeros_like(word_idx))
            dy_idx = torch.where(c_, torch.zeros_like(word_idx), word_idx - b_)
            st_emb = self._st_emb_layer(st_idx)
            dy_emb = self._dy_emb_layer(dy_idx)
            static_emb = torch.where(c_.unsqueeze(-1), st_emb, dy_emb)
            bottom_existence = torch.ones_like(word_idx, dtype = torch.bool)
        else:
            emb_layer = self._st_emb_layer or self._dy_emb_layer
            static_emb = emb_layer(word_idx)
            bottom_existence = word_idx > 0

        static_emb = self._dp_layer(static_emb)
        if self._act_pre_trained is not None:
            static_emb = self._act_pre_trained(static_emb)
        return static_emb, bottom_existence

contextual_config = dict(num_layers = num_ctx_layer, rnn_type = contextual_type, rnn_drop_out = frac_2)
class Contextual(nn.Module):
    def __init__(self,
                 model_dim,
                 num_layers,
                 rnn_type,
                 rnn_drop_out):
        super().__init__()
        if num_layers:
            assert model_dim % 2 == 0
            rnn_drop_out = rnn_drop_out if num_layers > 1 else 0
            self._contextual = rnn_type(model_dim,
                                        model_dim // 2,
                                        num_layers,
                                        bidirectional = True,
                                        batch_first = True,
                                        dropout = rnn_drop_out)
        else:
            self._contextual = None

    def forward(self, static_emb):
        if self._contextual is None:
            dynamic_emb = None
        else:
            dynamic_emb, _ = self._contextual(static_emb)
            dynamic_emb = dynamic_emb + static_emb # += does bad to gpu
            # dynamic_emb = self._dp_layer(dynamic_emb)
        return dynamic_emb