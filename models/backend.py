import torch
from torch import nn, Tensor
from utils.math_ops import s_index

from utils.types import BaseType, true_type, frac_4, frac_2
from utils.types import orient_dim, num_ori_layer

def valid_codebook(name):
    if name.startswith('codebook'):
        if '|' in name:
            bar = name.index('|') + 1
            try:
                bar = float(name[bar:])
            except:
                return False
        return bar >= 0
    return False

logit_type = BaseType('affine', default_set = ('affine', 'linear', 'codebook'), validator = valid_codebook)
contextual_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.LSTM, nn.GRU))
activation_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.ReLU, nn.ReLU6, nn.Softplus,# end == 0, nn.GELU
                                                                                   nn.LeakyReLU, nn.ELU, nn.CELU, nn.SELU, nn.RReLU, # end < 0
                                                                                   nn.Sigmoid, nn.LogSigmoid,
                                                                                   nn.Tanh, nn.Softsign, nn.Hardtanh, # -<0<+
                                                                                   nn.Tanhshrink, nn.Softshrink, nn.Hardshrink)) # -0+

from models.combine import get_combinator, combine_type
stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   num_layers   = num_ori_layer,
                   rnn_drop_out = frac_2,
                   drop_out     = frac_4,
                   trainable_initials = true_type)

from models.utils import condense_helper, condense_left
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
            orient = self.predict_orient(unit_hidden, h0c0)
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

        return (layers_of_unit, layers_of_existence, layers_of_orient, (segment, seg_length))


from models.utils import PCA
from utils.types import false_type, num_ctx_layer
from utils.param_ops import HParams, dict_print
act_fasttext = BaseType(None, as_index = True, as_exception = True, default_set = (nn.Tanh, nn.Softsign))
input_config = dict(pre_trained = true_type, activation = act_fasttext, trainable = false_type, drop_out = frac_4)

class InputLeaves(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tokens,
                 initial_weight,
                 pre_trained,
                 trainable,
                 activation,
                 drop_out):
        super().__init__()

        fixed_mutable_bound = 0
        fixed_emb_layer = mutable_emb_layer = None
        if pre_trained:
            num_special_tokens = num_tokens - initial_weight.shape[0]
            assert num_special_tokens >= 0
            # import pdb; pdb.set_trace()
            if num_special_tokens > 0: # bos eos
                if trainable:
                    initial_weight = torch.cat([torch.tensor(initial_weight), torch.rand(num_special_tokens, model_dim)], 0)
                    mutable_emb_layer = nn.Embedding.from_pretrained(initial_weight)
                else:
                    assert model_dim == initial_weight.shape[1]
                    fixed_mutable_bound = initial_weight.shape[0]
                    fixed_emb_layer     = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
                    mutable_emb_layer   = nn.Embedding(num_special_tokens, model_dim)
            elif trainable: # nil nil
                mutable_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
            else: # nil nil
                fixed_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
        else: # nil ... unk | ... unk bos eos
            assert trainable
            mutable_emb_layer = nn.Embedding(num_tokens, model_dim)

        if activation is None:
            self._act_pre_trained = None
        else:
            self._act_pre_trained = activation()
        self._dp_layer = nn.Dropout(drop_out)
        self._model_dim = model_dim
        self._fixed_mutable_bound = fixed_mutable_bound
        self._fixed_emb_layer = fixed_emb_layer
        self._mutable_emb_layer = mutable_emb_layer
        self._pca_base = None

    @property
    def has_only_dynamic(self):
        return self._fixed_emb_layer is None

    def flush_pc(self):
        self._pca_base = PCA(self._mutable_emb_layer.weight)

    def pca(self, word_emb):
        if self._fixed_emb_layer is not None and self._pca_base is None:
            self._pca_base = PCA(self._fixed_emb_layer.weight)
        return self._pca_base(word_emb)

    def forward(self, word_idx):
        # import pdb; pdb.set_trace()
        if self._fixed_mutable_bound > 0: # [nil] vocab | UNK | BOS EOS
            b_ = self._fixed_mutable_bound
            c_ = word_idx < b_
            f0_idx = torch.where(c_, word_idx, torch.zeros_like(word_idx))
            fb_idx = torch.where(c_, torch.zeros_like(word_idx), word_idx - b_)
            f0_emb = self._fixed_emb_layer(f0_idx)
            fb_emb = self._mutable_emb_layer(fb_idx)
            static_emb = torch.where(c_.unsqueeze(-1), f0_emb, fb_emb)
            bottom_existence = torch.ones_like(word_idx, dtype = torch.bool)
        else:
            emb_layer = self._fixed_emb_layer or self._mutable_emb_layer
            static_emb = emb_layer(word_idx)
            bottom_existence = word_idx > 0

        static_emb = self._dp_layer(static_emb)
        if self._act_pre_trained is not None:
            static_emb = self._act_pre_trained(static_emb)
        return static_emb, bottom_existence

state_usage = BaseType(None, as_index = False, as_exception = True,
                       validator   = lambda x: isinstance(x, int),
                       default_set = ('sum_layers', 'weight_layers'))

contextual_config = dict(num_layers   = num_ctx_layer,
                         rnn_type     = contextual_type,
                         rnn_drop_out = frac_2,
                         use_state    = dict(from_cell = true_type, usage = state_usage))

class Contextual(nn.Module):
    def __init__(self,
                 model_dim,
                 hidden_dim,
                 num_layers,
                 rnn_type,
                 rnn_drop_out,
                 use_state,
                 ):
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
            state_none_num_sum_weight = use_state['usage']
            use_cell_as_state = use_state['from_cell']
            if state_none_num_sum_weight is None:
                self._state_config = None
            else:
                if use_cell_as_state:
                    assert rnn_type is nn.LSTM, 'GRU does not have a cell'
                if state_none_num_sum_weight == 'weight_layers':
                    self._layer_weights = nn.Parameter(torch.zeros(num_layers, 2, 1, 1))
                    self._layer_softmax = nn.Softmax(dim = 0)
                    self._state_to_top3 = nn.Linear(model_dim, 3 * hidden_dim)
                self._state_config = num_layers, use_cell_as_state, state_none_num_sum_weight, hidden_dim
        else:
            self._contextual = None

    def forward(self, static_emb):
        if self._contextual is None:
            dynamic_emb = final_state = None
        else:
            dynamic_emb, final_state = self._contextual(static_emb)
            dynamic_emb = dynamic_emb + static_emb # += does bad to gpu
            if self._state_config is None:
                final_state = None
            else:
                num_layers, use_cell_as_state, state_none_num_sum_weight, hidden_dim = self._state_config
                if isinstance(final_state, tuple):
                    final_state = final_state[use_cell_as_state]
                batch_size, _, model_dim = dynamic_emb.shape
                final_state = final_state.view(num_layers, 2, batch_size, model_dim // 2)
                if isinstance(state_none_num_sum_weight, int): # some spec layer
                    final_state = final_state[state_none_num_sum_weight]
                else: # sum dim = 0
                    if state_none_num_sum_weight == 'weight_layers':
                        layer_weights = self._layer_softmax(self._layer_weights)
                        final_state = final_state * layer_weights
                    final_state = final_state.sum(dim = 0)
                # final_state: [batch, model_dim]
                final_state = final_state.transpose(0, 1).reshape(batch_size, model_dim)
                if use_cell_as_state:
                    final_state = torch.tanh(final_state)
                final_state = self._state_to_top3(final_state).reshape(batch_size, 3, hidden_dim)
        return dynamic_emb, final_state
