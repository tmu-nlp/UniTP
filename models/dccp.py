import torch
from torch import nn, Tensor
from collections import namedtuple
from models.types import activation_type, logit_type
from utils.types import orient_dim, hidden_dim, num_ori_layer, true_type, frac_2, frac_4, frac_5, BaseWrapper, BaseType

DiscoThresholds = namedtuple('DiscoThresholds', 'orient, joint, direct')

def disco_orient_to_dict(x):
    comps = {}
    for m_c in x.split(';'):
        m, c = m_c.split('.')
        for ci in c:
            comps[ci] = m
    return comps

class DiscoOrient(BaseWrapper):
    def __init__(self, name):
        super().__init__(name, lambda x: x)
        self._type = disco_orient_to_dict(self.name)

    def identical(self, x):
        return self._type == disco_orient_to_dict(x)

    # def __getitem__(self, comp):
    #     return self._type[comp]

disco_orient_type = []
for i in range(4 + 8 + 2):
    hinge = []
    crsen = []
    if i < 12:
        (crsen if i & 1 else hinge).append('j')
        (crsen if i & 2 else hinge).append('r')
        if i < 8:
            (crsen if i & 4 else hinge).append('d')
        if hinge:
            do_type = 'hinge.' + ''.join(hinge)
            if crsen: do_type += ';'
        else:
            do_type = ''
        if crsen:
            do_type += 'bce.' + ''.join(crsen)
    else:
        do_type = ('hinge' if i & 1 else 'bce') + '.j;ce.dr'
    disco_orient_type.append(DiscoOrient(do_type))
disco_orient_type = BaseType(-1, as_index = True, as_exception = True, default_set = disco_orient_type)

'i:add/cat:200:act|1'
'i:add|1'
def parse_joint_type(x):
    x = x.split(':')
    n = len(x)
    if n == 1:
        if x[0] in ('add', 'iadd'):
            return x
    elif n == 3:
        out, size, act = x
        size = int(size)
        if out not in ('add', 'iadd', 'cat', 'icat'):
            return False
        if out in ('cat', 'icat') and (size % 2 or size < 2):
            return False
        if activation_type.validate(act):
            return out, size, activation_type[act]
    elif n == 5:
        out, size_1, act_1, size_2, act_2 = x
        size_1 = int(size_1)
        size_2 = int(size_2)
        if out not in ('add', 'iadd', 'cat', 'icat'):
            return False
        if out in ('cat', 'icat') and (size_1 % 2 or size_1 < 2):
            return False
        if activation_type.validate(act_1) and activation_type.validate(act_2) and size_2 > 1:
            return out, size_1, activation_type[act_1], size_2, activation_type[act_2]
    return False
joint_type = BaseType('iadd', validator = parse_joint_type)

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

def convert32(left_undirec_right, right_only = False):
    left_undirec_right = torch.exp(left_undirec_right)
    right = left_undirec_right[:, :, 2]
    direc = left_undirec_right[:, :, 1]  + right
    right = right / direc
    if right_only:
        return right
    direc = direc / (direc + 2 * left_undirec_right[:, :, 0]) # this is a 2 to 1 battle, make it fair
    return right, direc # all in sigmoid range

def convert23_gold(right, direc):
    right = 1 + right
    return torch.where(direc, right, torch.zeros_like(right))

from models.types import SAL, LSA, orient_module, hinge_bias
from models.combine import get_combinator, get_components, combine_type, valid_trans_compound
stem_config = dict(orient_dim    = orient_dim,
                   combine_type  = combine_type,
                   orient_module = orient_module,
                   orient_type   = disco_orient_type,
                   joint_type    = joint_type,
                   local_joint   = true_type,
                   threshold     = dict(right = frac_5, direc = frac_4, joint = frac_4),
                   num_layers    = num_ori_layer,
                   rnn_drop_out  = frac_2,
                   drop_out      = frac_4,
                   trainable_initials = true_type)
from models.loss import cross_entropy, hinge_loss, binary_cross_entropy
from models.utils import hinge_score as hinge_score_
from models import StemOutput

class DiscoStem(nn.Module):
    def __init__(self,
                 model_dim,
                 orient_dim,
                 combine_type,
                 joint_type,
                 local_joint,
                 threshold,
                 orient_module,
                 num_layers,
                 orient_type,
                 rnn_drop_out,
                 trainable_initials,
                 drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        hinge_score = lambda x: hinge_score_(x, False)
        self._orient_type = orient_type = disco_orient_to_dict(orient_type)
        self._jnt_score, joint_loss = (hinge_score, hinge_loss) if orient_type['j'] == 'hinge' else (nn.Sigmoid(), binary_cross_entropy)
        if 'ce' in orient_type.values():
            assert orient_type.keys() == set('jdr')
            orient_bits = 3
            self._orient_scores = convert32
            self._loss_fns = joint_loss
        else:
            right_act, right_loss = (hinge_score, hinge_loss) if orient_type['r'] == 'hinge' else (nn.Sigmoid(), binary_cross_entropy)
            with_direc = 'd' in orient_type.keys()
            orient_bits = 1 + with_direc
            if with_direc:
                direc_act, direc_loss = (hinge_score, hinge_loss) if orient_type['d'] == 'hinge' else (nn.Sigmoid(), binary_cross_entropy)
            else:
                direc_act = direc_loss = None
            self._rgt_act = right_act
            self._dir_act = direc_act
            def orient_scores(right_direc):
                right = right_act(right_direc[:, :, 0])
                direc = direc_act(right_direc[:, :, 1]) if with_direc else None
                return right, direc
            self._orient_scores = orient_scores
            self._loss_fns = right_loss, direc_loss, joint_loss
        # bias_fns = DiscoThresholds(right_inv, joint_inv, direc_inv)
        self._raw_threshold = raw_threshold = DiscoThresholds(**threshold)
        assert 0 < raw_threshold.joint < 1
        assert 0 < raw_threshold.direc < 1
        self._orient_bits = orient_bits
        # self._thresholds = DiscoThresholds(*(fn(x) for fn, x in zip(bias_fns, raw_threshold)))
        
        self._is_sa = is_sa = orient_module in (SAL, LSA)
        if is_sa:
            self._orient_emb = orient_module(model_dim, orient_dim, num_layers)
        else:
            self._orient_emb = orient_module(model_dim, hidden_size,
                                             num_layers    = num_layers,
                                             bidirectional = True,
                                             batch_first   = True,
                                             dropout = rnn_drop_out if num_layers > 1 else 0)
        # self._orient_emb = nn.Linear(model_dim, orient_dim)
        self._stem_dp = dp_layer = nn.Dropout(drop_out)
        self._ori_dir = nn.Linear(orient_dim, orient_bits)
        joint_type = parse_joint_type(joint_type)
        jnt_n = len(joint_type)
        joint_in_size = model_dim if local_joint else orient_dim
        if jnt_n == 1:
            self._jnt_lhs = jnt_lhs = nn.Linear(joint_in_size, 1)
            self._jnt_rhs = jnt_rhs = nn.Linear(joint_in_size, 1, bias = False) if joint_type[0] != 'iadd' else self._jnt_lhs
            def jnt_fn(x):
                x = jnt_lhs(x[:, :-1]) + jnt_rhs(x[:, 1:])
                return x.squeeze(dim = 2)
        else:
            if jnt_n == 3:
                out, jnt_size, jnt_act = joint_type
            else:
                out, jnt_size, jnt_act, jnt_sec_size, jnt_sec_act = joint_type
            is_cat = out.endswith('cat')
            hid_size = (jnt_size >> 1) if is_cat else jnt_size
            self._jnt_lhs = jnt_lhs = nn.Linear(joint_in_size, hid_size)
            self._jnt_rhs = jnt_rhs = nn.Linear(joint_in_size, hid_size, bias = False) if out[0] != 'i' else jnt_lhs
            self._jnt_act = jnt_act = jnt_act()
            if jnt_n == 3:
                self._jnt_lgt = jnt_lgt = nn.Linear(jnt_size, 1)
                if is_cat:
                    def jnt_fn(x):
                        x = torch.cat([jnt_lhs(x[:, :-1]), jnt_rhs(x[:, 1:])], dim = 2)
                        x = dp_layer(x)
                        return jnt_lgt(jnt_act(x)).squeeze(dim = 2)
                else:
                    def jnt_fn(x):
                        x = jnt_lhs(x[:, :-1]) + jnt_rhs(x[:, 1:])
                        x = dp_layer(x)
                        return jnt_lgt(jnt_act(x)).squeeze(dim = 2)
            else:
                self._jnt_sec_hid = jnt_sec_hid = nn.Linear(jnt_size, jnt_sec_size)
                self._jnt_sec_act = jnt_sec_act = jnt_sec_act()
                self._jnt_lgt = jnt_lgt = nn.Linear(jnt_sec_size, 1)
                if is_cat:
                    def jnt_fn(x):
                        x = torch.cat([jnt_lhs(x[:, :-1]), jnt_rhs(x[:, 1:])], dim = 2)
                        x = dp_layer(x)
                        x = jnt_sec_hid(jnt_act(x))
                        x = dp_layer(x)
                        return jnt_lgt(jnt_sec_act(x)).squeeze(dim = 2)
                else:
                    def jnt_fn(x):
                        x = jnt_lhs(x[:, :-1]) + jnt_rhs(x[:, 1:])
                        x = dp_layer(x)
                        x = jnt_sec_hid(jnt_act(x))
                        x = dp_layer(x)
                        return jnt_lgt(jnt_sec_act(x)).squeeze(dim = 2)

        self._jnt_fn_local_flag = jnt_fn, local_joint
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

    def predict_orient_direc(self, unit_hidden, h0c0, seq_len):
        if self._is_sa:
            orient_hidden = self._orient_emb(unit_hidden, seq_len)
        else:
            orient_hidden, _ = self._orient_emb(unit_hidden, h0c0)
        # orient_hidden = self._orient_emb(unit_hidden) #
        orient_hidden = self._stem_dp(orient_hidden)
        return self._ori_dir(orient_hidden), orient_hidden
    
    def forward(self,
                existence,
                unit_emb,
                swap = None,
                supervised_right = None,
                supervised_joint = None,
                **kw_args):
        batch_size, seg_len = existence.shape
        max_iter_n = seg_len << 1 # 2 times
        h0c0 = self.get_h0c0(batch_size)
        # existence.squeeze_(dim = 2) # in-place is a pandora box

        layers_of_joint = []
        layers_of_u_emb = []
        layers_of_right_direc = []
        layers_of_existence = []
        layers_of_shuffled_right_direc = []
        layers_of_shuffled_joint = []
        teacher_forcing = isinstance(supervised_right, Tensor)
        if teacher_forcing:
            assert isinstance(supervised_joint, Tensor)
            ori_start = 0
            jnt_start = 0
        segment, seg_length = [], []
        history = []
        jnt_fn, local_joint = self._jnt_fn_local_flag

        for l_cnt in range(max_iter_n): # max_iter | max_tree_high (unit_emb ** 2).sum().backward()
            # existence = unit_idx > 0
            seq_len = existence.sum(dim = 1)
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(seq_len)

            right_direc, rd_hidden = self.predict_orient_direc(unit_emb, h0c0, seq_len)
            layers_of_u_emb.append(unit_emb)
            layers_of_right_direc.append(right_direc)
            layers_of_existence.append(existence)
            if teacher_forcing and swap is not None:
                if l_cnt == 0:# or random() < 0.1:
                    base = torch.zeros_like(unit_emb)
                    base = base.scatter(1, swap[l_cnt].unsqueeze(2).expand_as(unit_emb), unit_emb)
                shuffled_right_direc, shuffled_rd_hidden = self.predict_orient_direc(base, h0c0, seq_len)
                layers_of_shuffled_right_direc.append(shuffled_right_direc)
            if seg_len == 2:
                break # teacher forcing or a good model
            elif len(history) > 1:
                if len(history) > 2:
                    pprev, prev, curr = history
                else:
                    pprev = None
                    prev, curr = history
                if prev.shape == curr.shape and (prev == curr).all() or pprev is not None and pprev.shape == curr.shape and (pprev == curr).all():
                    break
                elif l_cnt == max_iter_n - 1:
                    from sys import stderr
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break
            joint = jnt_fn(unit_emb if local_joint else rd_hidden)
            layers_of_joint.append(joint)

            if teacher_forcing:
                ori_end   = ori_start + seg_len
                jnt_end   = jnt_start + seg_len - 1
                right = supervised_right[:, ori_start:ori_end]
                joint = supervised_joint[:, jnt_start:jnt_end]
                ori_start = ori_end
                jnt_start = jnt_end
                direc = None # not important for

                if swap is not None:
                    layers_of_shuffled_joint.append(jnt_fn(base if local_joint else shuffled_rd_hidden))
                    lhs, rhs, jnt, _ = split(base, right, joint, existence, direc)
                    base = self.combine(lhs, rhs, jnt.unsqueeze(dim = 2))
            else:
                right, joint, direc = self.get_stem_prediction(right_direc, joint)

            lhs, rhs, jnt, ids = split(unit_emb, right, joint, existence, direc)
            unit_emb = self.combine(lhs, rhs, jnt.unsqueeze(dim = 2))
            seg_len  = unit_emb.shape[1]
            existence = ids > 0

            if not teacher_forcing:
                history.append(ids)
                if len(history) > 3:
                    history.pop(0)

        embeddings  = torch.cat(layers_of_u_emb,       dim = 1)
        right_direc = torch.cat(layers_of_right_direc, dim = 1)
        joint       = torch.cat(layers_of_joint,       dim = 1)
        existence   = torch.cat(layers_of_existence,   dim = 1)

        shuffled_right_direc = shuffled_joint = None
        if teacher_forcing:
            if swap is not None:
                shuffled_right_direc = torch.cat(layers_of_shuffled_right_direc, dim = 1)
                shuffled_joint       = torch.cat(layers_of_shuffled_joint,       dim = 1)
            assert joint.shape[1] == supervised_joint.shape[1], f'{joint.shape[1]} vs. {supervised_joint.shape[1]}'
            assert right_direc.shape[1] == supervised_right.shape[1], f'{right_direc.shape[1]} vs. {supervised_right.shape[1]}'
        else:
            # segment    = torch.stack(segment,    dim = 0)
            seg_length = torch.stack(seg_length, dim = 1)

        return StemOutput(embeddings, existence, segment, (right_direc, joint, shuffled_right_direc, shuffled_joint, seg_length))

    @property
    def orient_bits(self):
        return self._orient_bits

    def get_stem_score(self, right_direc, joint):
        right_score, direc_score = self._orient_scores(right_direc)
        joint_score = self._jnt_score(joint)
        return right_score, joint_score, direc_score

    def get_stem_prediction(self, right_direc, joint, get_score = False):
        right_score, joint_score, direc_score = self.get_stem_score(right_direc, joint)

        right = right_score > self._raw_threshold.right
        direc = direc_score > self._raw_threshold.direc if direc_score is not None else None
        joint = joint_score > self._raw_threshold.joint
        if get_score:
            return right, joint, direc, right_score, joint_score, direc_score
        return right, joint, direc

    def get_stem_loss(self, gold, right_direc_logits, joint_logits, undirec_strength):
        gold_right = gold['right']
        gold_direc = gold['direc']
        gold_joint = gold['joint']
        if (orient_bits := self._orient_bits) == 3:
            orient_loss = cross_entropy(right_direc_logits, convert23_gold(gold_right, gold_direc))
            return orient_loss, self._loss_fns(joint_logits, gold_joint, None)

        if orient_bits == 2:
            if undirec_strength == 0:
                direc_weight = gold_direc
            elif undirec_strength == 1:
                direc_weight = gold['existence']
            else:
                direc_weight = gold_direc * (1 - undirec_strength) + undirec_strength
                direc_weight *= gold['existence']
        else:
            direc_weight = gold_direc
        right_loss, direc_loss, joint_loss = self._loss_fns
        if direc_loss is not None:
            direc_loss = direc_loss(right_direc_logits[:, :, 1], gold_direc, None)
        right_loss = right_loss(right_direc_logits[:, :, 0], gold_right, direc_weight)
        joint_loss = joint_loss(joint_logits, gold_joint, None)

        return right_loss, joint_loss, direc_loss


multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(orient_layer    = stem_config,
                  tag_label_layer = multi_class)

from models.loss import get_loss
from models.backend import OutputLayer
from utils.param_ops import change_key
class BaseRnnParser(OutputLayer):
    def __init__(self, *args, **kwargs):
        change_key(kwargs, 'orient_layer', 'stem_layer')
        super().__init__(DiscoStem, *args, **kwargs)

    def get_losses(self, batch, weight_mask, tag_logits, top3_label_logits, label_logits, key = None):
        tag_fn = self._tag_layer if key is None else self._tag_layer[key]
        label_fn = self._label_layer if key is None else self._label_layer[key]
        height_mask = batch['segments'][None] * (batch['seq_len'] > 0)
        height_mask = height_mask.sum(dim = 1)
        tag_loss   = get_loss(tag_fn,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(label_fn, self._logit_max, label_logits, batch, False, height_mask, weight_mask, 'label')
        if top3_label_logits is not None:
            label_loss += get_loss(label_fn, self._logit_max, top3_label_logits, batch, 'top3_label')
        return tag_loss, label_loss