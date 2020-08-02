import torch
from torch import nn, Tensor
from models.backend import Stem, activation_type, logit_type

from utils.types import orient_dim, hidden_dim, num_ori_layer, true_type, frac_2, frac_4

from models.combine import get_combinator, get_components, combine_type, valid_trans_compound
stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   joint_act    = activation_type,
                   num_layers   = num_ori_layer,
                   rnn_drop_out = frac_2,
                   drop_out     = frac_4,
                   trainable_initials = true_type)

from models.utils import condense_splitter, condense_left
def diff_integer_indice(right_layer, joint_layer, existence, direc_layer = None, test = False):
    lhs_helper, rhs_helper, _, swap, bool_pads = condense_splitter(right_layer, joint_layer, existence)
    if direc_layer is not None:
        swap &= direc_layer[:, 1:] | direc_layer[:, :-1]
    lhs_diff = torch.cat([swap, bool_pads, bool_pads], dim = 1)
    mid_diff = torch.cat([bool_pads, swap, bool_pads], dim = 1)
    rhs_diff = torch.cat([bool_pads, bool_pads, swap], dim = 1)
    diff_indices = 1 * lhs_diff - 2 * mid_diff + 1 * rhs_diff
    # assert diff_indices.sum() == 0 # TODO: CHECKED
    diff_indices = diff_indices[:, :-1] * existence + existence
    if test:
        return diff_indices
    return lhs_helper, rhs_helper, torch.cumsum(diff_indices, dim = 1)

def split(hidden, right_layer, joint_layer, existence, direc_layer = None):
    lhs_helper, rhs_helper, indices = diff_integer_indice(right_layer, joint_layer, existence, direc_layer)
    lhs_indices = condense_left(indices, lhs_helper, skip_dump0 = False)
    rhs_indices = condense_left(indices, rhs_helper, skip_dump0 = False)
    indices = torch.arange(hidden.shape[0], device = indices.device)[:, None]
    lhs_indices[:, 0] = 0
    rhs_indices[:, 0] = 0
    lhs_hidden = hidden[indices, lhs_indices]
    rhs_hidden = hidden[indices, rhs_indices]
    return lhs_hidden, rhs_hidden, rhs_indices > 0, lhs_indices + rhs_indices

class DiscoStem(nn.Module):
    def __init__(self,
                 model_dim,
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
                unit_emb,
                existence,
                supervised_right = None,
                supervised_joint = None,
                **kw_args):
        batch_size, seg_len = existence.shape
        max_iter_n = seg_len << 2 # 4 times
        h0c0 = self.get_h0c0(batch_size)
        # existence.squeeze_(dim = 2) # in-place is a pandora box

        layers_of_joint = []
        layers_of_u_emb = []
        layers_of_right_direc = []
        layers_of_existence = []
        teacher_forcing = isinstance(supervised_right, Tensor)
        if teacher_forcing:
            assert isinstance(supervised_joint, Tensor)
            ori_start = 0
            jnt_start = 0
        segment, seg_length = [], []
        history = []

        for l_cnt in range(max_iter_n): # max_iter | max_tree_high (unit_emb ** 2).sum().backward()
            # existence = unit_idx > 0
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(existence.sum(dim = 1))

            right_direc = self.predict_orient_direc(unit_emb, h0c0)
            layers_of_u_emb.append(unit_emb)
            layers_of_right_direc.append(right_direc)
            layers_of_existence.append(existence)
            if seg_len == 2:
                break # teacher forcing or a good model
            elif len(history) > 1:
                prev, curr = history
                if prev.shape == curr.shape and (prev == curr).all():
                    break
            joint = self.predict_joint(unit_emb)
            layers_of_joint.append(joint)

            if teacher_forcing:
                ori_end   = ori_start + seg_len
                jnt_end   = jnt_start + seg_len - 1
                right = supervised_right[:, ori_start:ori_end]
                joint = supervised_joint[:, jnt_start:jnt_end]
                ori_start = ori_end
                jnt_start = jnt_end
                direc = None
            else:
                right = right_direc[:, :, 0] > 0
                direc = right_direc[:, :, 1] > 0
                joint = joint > 0

            lhs, rhs, jnt, ids = split(unit_emb, right, joint, existence, direc)
            unit_emb = self.combine(lhs, rhs, jnt.unsqueeze(dim = 2))
            seg_len  = unit_emb.shape[1]
            existence = ids > 0

            if not teacher_forcing:
                history.append(ids)
                if len(history) > 2:
                    history.pop(0)

            if l_cnt == max_iter_n - 1: print('Unknown action')

        embeddings  = torch.cat(layers_of_u_emb,       dim = 1)
        right_direc = torch.cat(layers_of_right_direc, dim = 1)
        joint       = torch.cat(layers_of_joint,       dim = 1)
        existence   = torch.cat(layers_of_existence,   dim = 1)

        if teacher_forcing:
            assert joint.shape[1] == supervised_joint.shape[1]
            assert right_direc.shape[1] == supervised_right.shape[1]
        else:
            # segment    = torch.stack(segment,    dim = 0)
            seg_length = torch.stack(seg_length, dim = 1)

        return existence, embeddings, right_direc, joint, segment, seg_length

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(orient_layer    = stem_config,
                  tag_label_layer = multi_class)
from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value, get_loss

class BaseRnnTree(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 orient_layer,
                 tag_label_layer,
                 **kw_args):
        super().__init__(**kw_args)

        self._stem_layer = DiscoStem(model_dim, **orient_layer)

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
        (layers_of_existence, layers_of_base, layers_of_right_direc, layers_of_joint, segment,
         seg_length) = self._stem_layer(base_inputs, bottom_existence, **kw_args)

        if self._hidden_dim:
            layers_of_hidden = self._shared_layer(layers_of_base)
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

        return layers_of_existence, layers_of_base, layers_of_hidden, layers_of_right_direc, layers_of_joint, tags, labels, segment, seg_length

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

    def get_losses(self, batch, tag_logits, top3_label_logits, label_logits):
        height_mask = batch['segments'][None] * (batch['seq_len'] > 0)
        height_mask = height_mask.sum(dim = 1)
        tag_loss   = get_loss(self._tag_layer,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(self._label_layer, self._logit_max, label_logits, batch, False, height_mask, 'label')
        if top3_label_logits is not None:
            tag_loss += get_loss(self._label_layer, self._logit_max, top3_label_logits, batch, 'top3_label')
        return tag_loss, label_loss