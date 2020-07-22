import torch
from torch import nn, Tensor
from models.backend import Stem, activation_type, logit_type

from utils.types import hidden_dim, frac_4

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

from models.backend import stem_config
penn_tree_config = dict(orient_layer    = stem_config,
                        tag_label_layer = multi_class)
from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value, get_loss

class DiscoStem(nn.Module):
    def __init__(self,
                 model_dim,
                 index_cnn,
                 orient_dim,
                 combine_type,
                 joint_act,
                 num_layers,
                 rnn_drop_out,
                 trainable_initials,
                 drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        self._orient_emb = nn.LSTM(model_dim, hidden_size,
                                   num_layers    = num_layers,
                                   bidirectional = True,
                                   batch_first   = True,
                                   dropout = rnn_drop_out if num_layers > 1 else 0)
        self._dp_layer = nn.Dropout(drop_out)
        self._ori_dir = nn.Linear(orient_dim, 2)
        self._jnt_bse = nn.Conv1d(model_dim, orient_dim, 2, 1, 0)
        self._jnt_act = joint_act()
        self._jnt_lgt = nn.Linear(orient_dim, 1)
        self._idx_cnn = index_cnn
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

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0_act(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def predict_orient_direc(self, unit_hidden, h0c0):
        orient_hidden, _ = self._orient_emb(unit_hidden, h0c0)
        orient_hidden = self._dp_layer(orient_hidden)
        return self._ori_dir(orient_hidden)

    def predict_joint(self, unit_hidden):
        joint_hidden = self._jnt_bse(unit_hidden.transpose(1, 2))
        joint_hidden = self._dp_layer(joint_hidden)
        joint_hidden = self._jnt_act(joint_hidden)
        return self._jnt_lgt(joint_hidden.transpose(1, 2)).squeeze(dim = 2)
    
    def forward(self,
                unit_idx,
                unit_emb,
                supervised_right = None,
                supervised_joint = None,
                **kw_args):
        batch_size, seg_len = unit_idx.shape
        h0c0 = self.get_h0c0(batch_size)

        layers_of_joint = []
        layers_of_u_emb = []
        layers_of_right_direc = []
        teacher_forcing = isinstance(supervised_right, Tensor)
        if teacher_forcing:
            ori_start = 0
            jnt_start = 0
        segment, seg_length = [], []
        last_unit_idx = None

        for _ in range(50): # max_iter | max_tree_high
            existence = unit_idx > 0
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(existence.sum(dim = 1))

            right_direc = self.predict_orient_direc(unit_emb, h0c0)
            joint = self.predict_joint(unit_emb)
            layers_of_joint.append(joint)
            layers_of_u_emb.append(unit_emb)
            layers_of_right_direc.append(right_direc)
            if seg_len == 2 or last_unit_idx is not None and last_unit_idx.shape == unit_idx.shape and (last_unit_idx == unit_idx).all(): break

            if teacher_forcing:
                ori_end   = ori_start + seg_len
                jnt_end   = jnt_start + seg_len - 1
                right = supervised_right[:, ori_start:ori_end]
                joint = supervised_joint[:, jnt_start:jnt_end]
                ori_start = ori_end
                jnt_start = jnt_end
            else:
                right = right_direc[:, :, 0] > 0
                joint = joint > 0
            right.unsqueeze_(dim = 2)

            lhs, rhs, jnt, ids = self._idx_cnn.split(self, unit_emb, right, joint, existence)
            unit_emb = self.combine(lhs, rhs, jnt)
            seg_len = unit_emb.shape[1]
            last_unit_idx = unit_idx
            unit_idx      = ids

        if not teacher_forcing:
            segment    = torch.cat(segment,    dim = 1)
            seg_length = torch.cat(seg_length, dim = 1)

        embeddings  = torch.cat(layers_of_u_emb,       dim = 1)
        right_direc = torch.cat(layers_of_right_direc, dim = 1)
        joint       = torch.cat(layers_of_joint,       dim = 1)

        return embeddings, right_direc, joint, segment, seg_length


class BaseRnnTree(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 orient_layer,
                 tag_label_layer,
                 **kw_args):
        super().__init__(**kw_args)

        self._stem_layer = Stem(model_dim, **orient_layer)

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

        (layers_of_base, layers_of_orient, layers_of_existence,
         trapezoid_info) = self._stem_layer(bottom_existence,
                                            base_inputs, # dynamic can be none
                                            **kw_args)

        if self._hidden_dim:
            layers_of_hidden = self._shared_layer(layers_of_base)
            layers_of_hidden = self._dp_layer(layers_of_hidden)
            if self._logit_max:
                layers_of_hidden = self._activation(layers_of_hidden)

            if self._tag_layer is None or ingore_logits:
                tags = None
            else:
                _, batch_len, _ = base_inputs.shape
                tags = self._tag_layer(layers_of_hidden[:, -batch_len:])
            
            if self._label_layer is None or ingore_logits:
                labels = None
            else:
                labels = self._label_layer(layers_of_hidden)
        else:
            layers_of_hidden = tags = labels = None

        return layers_of_base, layers_of_hidden, layers_of_existence, layers_of_orient, tags, labels, trapezoid_info

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

    def get_loss(self, logits, batch, height_mask):
        if height_mask is None:
            return get_loss(self._logit_max, logits, batch, self._tag_layer)
        return get_loss(self._logit_max, logits, batch, self._label_layer, height_mask, 'label')