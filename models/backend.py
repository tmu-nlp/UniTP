import torch
from torch import nn, Tensor
from utils.math_ops import s_index

from utils.types import BaseType, true_type, false_type, vocab_size
from utils.types import word_dim, orient_dim, num_pre_layer, frac_4, frac_2

contextual_type = BaseType(0, as_index = True, as_exception = True, default_set = (nn.LSTM, nn.GRU))

from models.utils import PCA

leaves_config = dict(use_fasttext = true_type,
                     word_dim     = word_dim,
                     trainable    = false_type,
                     contextual   = contextual_type,
                     num_layers   = num_pre_layer,
                     drop_out     = frac_4)

class Leaves(nn.Module):
    def __init__(self,
                 num_words,
                 initial_weight,
                 use_fasttext,
                 word_dim,
                 trainable,
                 contextual,
                 num_layers,
                 drop_out):
        super().__init__()

        # word
        st_dy_bound = 0
        st_emb_layer = dy_emb_layer = None
        if use_fasttext:
            num_special_tokens = num_words - initial_weight.shape[0]
            assert num_special_tokens >= 0
            if num_special_tokens > 0:
                if trainable:
                    initial_weight = np.concatenate([initial_weight, np.random.random(num_special_tokens, word_dim)])
                    st_emb_layer = None
                    dy_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight))
                else:
                    assert word_dim == initial_weight[1]
                    st_dy_bound = initial_weight.shape[0]
                    st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
                    dy_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(np.random.random(num_special_tokens, word_dim)))
            elif trainable:
                dy_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
            else:
                st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
        else:
            assert trainable
            dy_emb_layer = nn.Embedding(num_words, word_dim)

        self._word_dim = word_dim
        self._st_dy_bound = st_dy_bound
        self._st_emb_layer = st_emb_layer
        self._dy_emb_layer = dy_emb_layer
        self._dp_layer = nn.Dropout(drop_out)
        self._pca_base = None

        # contextual
        if num_layers:
            if contextual in (nn.LSTM, nn.GRU):
                assert word_dim % 2 == 0
                self._contextual = contextual(word_dim,
                                              word_dim // 2,
                                              num_layers,
                                              bidirectional = True,
                                              batch_first = True)
            else:
                raise NotImplementedError()
        else:
            self._contextual = None

    @property
    def embedding_dim(self):
        return self._word_dim

    def pca(self, word_emb, flush = False):
        # TODO: setup_pca with external
        if self._st_emb_layer is not None and self._pca_base is None:
            self._pca_base = PCA(self._st_emb_layer.weight)
        return self._pca_base(word_emb)

    def forward(self, is_training, word_idx): # TODO: zeros for <nil> paddings
        if self._st_dy_bound > 0:
            b_ = self._st_dy_bound
            c_ = word_idx < b_
            st_idx = torch.where(c_, word_idx, torch.zeros_like(word_idx))
            dy_idx = torch.where(c_, torch.zeros_like(word_idx), word_idx - b_)
            st_emb = self._st_emb_layer(st_idx)
            dy_emb = self._dy_emb_layer(dy_idx)
            static_emb = torch.where(c_, st_emb, dy_emb)
        else:
            emb_layer = self._st_emb_layer or self._dy_emb_layer
            static_emb = emb_layer(word_idx)

        if is_training:
            static_emb = self._dp_layer(static_emb)

        if self._contextual is None:
            dynamic_emb = None
        else:
            dynamic_emb, _ = self._contextual(static_emb)
            dynamic_emb = dynamic_emb + static_emb # += does bad to gpu

        return static_emb, dynamic_emb

from utils.types import combine_type, true_type
stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   drop_out     = frac_4,
                   rnn_drop_out = frac_2,
                   trainable_initials = true_type)

from models.combine import get_combinator
class Stem(nn.Module):
    def __init__(self,
                 word_dim,
                 orient_dim,
                 combine_type,
                 trainable_initials,
                 drop_out,
                 rnn_drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        self.orient_emb = nn.LSTM(word_dim, hidden_size,
                                  bidirectional = True,
                                  batch_first = True)
        self._dp_layer = nn.Dropout(drop_out)
        self.orient = nn.Linear(orient_dim, 1)
        self.combine = get_combinator(combine_type, word_dim)
        self._rnn_drop_out = rnn_drop_out
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
                supervised_orient):
        existence.unsqueeze_(dim = -1)
        unit_hidden *= existence
        batch_size, num_layers, _ = unit_hidden.shape
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0c0 = h0, c0
        else:
            h0c0 = None
        
        is_training = isinstance(supervised_orient, Tensor)
        modification = not is_training and isinstance(supervised_orient, tuple)
        if modification:
            offsets, lengths = supervised_orient
            batch_dim = torch.arange(batch_size, device = existence.device)
            ends = offsets + lengths - 1

        for length in range(num_layers, 0, -1):
            if is_training:
                self.orient_emb.dropout = self._rnn_drop_out
                orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
                orient_hidden = self._dp_layer(orient_hidden)
            else:
                self.orient_emb.dropout = 0
                orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
            orient = self.orient(orient_hidden)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if length == 1: break

            if supervised_orient is None:
                right = orient > 0
            elif modification:
                right = orient > 0
                starts = torch.where(offsets < length, offsets, torch.zeros_like(offsets))
                _ends_ = ends - (num_layers - length)
                _ends_ = torch.where( starts < _ends_,  _ends_, torch.ones_like(_ends_) * (length - 1))
                right[batch_dim, starts] = True
                right[batch_dim, _ends_] = False
            else:
                start = s_index(length - 1)
                end   = s_index(length)
                right = supervised_orient[:, start:end, None]
            
            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)

        layers_of_unit.reverse()
        layers_of_orient.reverse()
        layers_of_existence.reverse()

        unit_hidden = torch.cat(layers_of_unit,   dim = 1)
        orient      = torch.cat(layers_of_orient, dim = 1) # must be predicted
        existence   = torch.cat(layers_of_existence, dim = 1)

        return unit_hidden, orient, existence