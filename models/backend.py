import torch
from torch import nn
from utils.types import BaseType, true_type, frac_4, frac_2
from utils.types import num_ori_layer, false_type
from utils.types import num_ctx_layer, hidden_dim
from utils.param_ops import HParams
from models.types import act_fasttext
from models.utils import PCA
input_config = dict(pre_trained = true_type, activation = act_fasttext, drop_out = frac_4)

class InputLeaves(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tokens,
                 initial_weight,
                 nil_as_pad,
                #  unk_id,
                 pre_trained,
                 activation,
                 drop_out):#,
                #  random_unk_from_id,
                #  random_unk_prob):
        super().__init__()

        if initial_weight is None: # tokenization without <nil>, <bos> & <eos> are included tuned with others
            fixed_dim = model_dim
            assert not pre_trained
        else: # parsing / sentiment analysis
            # 0: no special; 1: <unk>; 2: <bos> <eos> (w/o <nil>); 3: 1 and 2 (w/o <nil>)
            fixed_num, fixed_dim = initial_weight.shape
            num_special_tokens   = num_tokens - fixed_num
            assert num_special_tokens >= 0

        main_extra_bound = 0
        main_emb_layer = extra_emb_layer = None
        padding_kwarg = dict(padding_idx = 0 if nil_as_pad else None)
        if pre_trained:
            if num_special_tokens > 0: # unk | bos + eos | no <nil>
                main_extra_bound = initial_weight.shape[0]
                main_emb_layer  = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
                extra_emb_layer = nn.Embedding(num_special_tokens, fixed_dim)
            else: # <nil> ... |
                main_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False, **padding_kwarg)
        else: # nil ... unk | ... unk bos eos
            main_emb_layer = nn.Embedding(num_tokens, fixed_dim, **padding_kwarg)
        static_pca = fixed_dim == model_dim

        if activation is None:
            self._act_pre_trained = None
        else:
            self._act_pre_trained = activation()
        
        self._dp_layer = nn.Dropout(drop_out)
        self._main_extra_bound_pad = main_extra_bound, nil_as_pad
        self._input_dim = fixed_dim
        self._main_emb_layer  = main_emb_layer
        self._extra_emb_layer = extra_emb_layer
        self._pca_base = None, static_pca
        self._main_emb_tuned = True

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def has_static_pca(self):
        return self._pca_base[1]

    def flush_pc_if_emb_is_tuned(self):
        pca_base, static = self._pca_base
        assert static, 'has_no_static_pca'
        if self._main_emb_tuned or pca_base is None:
            self._pca_base = PCA(self._main_emb_layer.weight), True
            self._main_emb_tuned = False

    def pca(self, word_emb):
        pca_base, static = self._pca_base
        assert static, 'has_no_static_pca'
        return pca_base(word_emb)

    def forward(self, word_idx, tune_pre_trained):
        bound, nil_as_pad = self._main_extra_bound_pad
        if bound > 0: # [nil] vocab | UNK | BOS EOS
            fix_mask = word_idx < bound
            f0_idx = fix_mask * word_idx
            fb_idx =~fix_mask * (word_idx - bound)
            if tune_pre_trained:
                f0_emb = self._main_emb_layer(f0_idx)
            else:
                with torch.no_grad():
                    f0_emb = self._main_emb_layer(f0_idx)
            fb_emb = self._extra_emb_layer(fb_idx) # UNK BOS EOS must be tuned
            static_emb = torch.where(fix_mask.unsqueeze(-1), f0_emb, fb_emb)
        else:
            emb_layer = self._main_emb_layer or self._extra_emb_layer
            if tune_pre_trained:
                static_emb = emb_layer(word_idx)
            else:
                with torch.no_grad():
                    static_emb = emb_layer(word_idx)
        if nil_as_pad:
            bottom_existence = word_idx > 0
            bottom_existence.unsqueeze_(dim = 2)
            # static_emb = static_emb * bottom_existence: done by padding_idx = 0
        else:
            bottom_existence = torch.ones_like(word_idx, dtype = torch.bool) # obsolete (only for nccp)
            bottom_existence.unsqueeze_(dim = 2)
        self._main_emb_tuned = self.training and tune_pre_trained

        static_emb = self._dp_layer(static_emb)
        if self._act_pre_trained is not None:
            static_emb = self._act_pre_trained(static_emb)
        return static_emb, bottom_existence


state_usage = BaseType(None, as_index = False, as_exception = True,
                       validator   = lambda x: isinstance(x, int),
                       default_set = ('sum_layers', 'weight_layers'))

from models.types import rnn_module_type
contextual_config = dict(num_layers   = num_ctx_layer,
                         rnn_type     = rnn_module_type,
                         rnn_drop_out = frac_2,
                         use_state    = dict(from_cell = true_type, usage = state_usage))

class Contextual(nn.Module):
    def __init__(self,
                 input_dim,
                 model_dim,
                 hidden_dim,
                 num_layers,
                 rnn_type,
                 rnn_drop_out,
                 use_state):
        super().__init__()
        if num_layers:
            assert input_dim % 2 == 0
            assert model_dim % 2 == 0
            rnn_drop_out = rnn_drop_out if num_layers > 1 else 0
            self._contextual = rnn_type(input_dim,
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

    @property
    def is_useless(self):
        return self._contextual is None

    def forward(self, static_emb):
        dynamic_emb, final_state = self._contextual(static_emb)

        if self._state_config is None:
            top_3 = None
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
            top_3 = self._state_to_top3(final_state).reshape(batch_size, 3, hidden_dim)

        return dynamic_emb, top_3


from models.combine import get_combinator
class InputLayer(nn.Module):
    def __init__(self,
                 paddings,
                 model_dim,
                 use,
                 word_emb,
                 char_rnn,
                 contextual_layer,
                 combine_static,
                 num_chars       = None,
                 num_tokens      = None,
                 initial_weights = None,
                 **kwargs_forwarding):
        super().__init__(model_dim, **kwargs_forwarding)
        if use['word_emb']:
            if isinstance(num_tokens, int):
                self._word_emb = InputLeaves(model_dim, num_tokens, initial_weights, not paddings, **word_emb)
                input_dim = self._word_emb.input_dim
            else:
                from utils.param_ops import get_sole_key
                self._word_emb = nn.ModuleDict({k: InputLeaves(model_dim, v, initial_weights[k], not paddings[k], **word_emb) for k,v in num_tokens.items()})
                input_dim = get_sole_key(n.input_dim for n in self._word_emb.values())
        else:
            self._word_emb = None
            input_dim = model_dim
        if use['char_rnn']:
            embed_dim = char_rnn['embed_dim']
            self._char_rnn = PadRNN(num_chars, None, None, fence_dim = model_dim, char_space_idx = 1, **char_rnn)
            self._char_lin = nn.Linear(embed_dim, model_dim)
            self._char_act = nn.Tanh()
        else:
            self._char_rnn = None

        contextual_layer = Contextual(input_dim, model_dim, self.hidden_dim, **contextual_layer)
        diff = model_dim - input_dim
        self._combine_static = None
        self._bias_only = False
        if contextual_layer.is_useless:
            self._contextual_layer = None
            assert diff == 0, 'useless difference'
        else:
            self._contextual_layer = contextual_layer
            if combine_static:
                self._bias_only = combine_static in ('NS', 'NV')
                self._combine_static = get_combinator(combine_static, input_dim)
            assert diff >= 0, 'invalid difference'
        self._half_dim_diff = diff >> 1

    def forward(self, word_idx, tune_pre_trained, ingore_logits = False,
                sub_idx = None, sub_fence = None, offset = None, key = None,
                squeeze_existence = None, 
                **kw_args):
        assert isinstance(squeeze_existence, bool)
        batch_size, batch_len = word_idx.shape
        if self._word_emb:
            emb = self._word_emb if key is None else self._word_emb[key]
            static, bottom_existence = emb(word_idx, tune_pre_trained)
            if self._char_rnn:
                char_info = self._char_rnn(sub_idx, sub_fence, offset)
                char_info = self._stem_dp(char_info)
                char_info = self._char_lin(char_info)
                char_info = self._char_act(char_info)
                static = static + char_info * bottom_existence
        else:
            bottom_existence = word_idx > 0
            bottom_existence.unsqueeze_(dim = 2)
            static = self._char_rnn(sub_idx, sub_fence, offset) * bottom_existence
        if self._contextual_layer is None:
            base_inputs = static
            top3_hidden = None
        else:
            dynamic, top3_hidden = self._contextual_layer(static)
            if self._half_dim_diff:
                zero_pads = torch.zeros(batch_size, batch_len, self._half_dim_diff, dtype = static.dtype, device = static.device)
                static = torch.cat([zero_pads, static, zero_pads], dim = 2)
            base_inputs  = dynamic * bottom_existence
            if self._combine_static is not None:
                base_inputs = self._combine_static.compose(static, base_inputs, None)
        if squeeze_existence:
            bottom_existence = bottom_existence.squeeze(dim = 2)
        base_returns = super().forward(base_inputs, bottom_existence, ingore_logits, key, **kw_args)
        top3_labels  = super().get_label(top3_hidden) if top3_hidden is not None else None
        return (batch_size, batch_len, static, top3_labels) + base_returns

    def get_static_pca(self, key = None):
        if self._word_emb:
            if isinstance(self._word_emb, InputLeaves):
                word_emb = self._word_emb
            else:
                word_emb = self._word_emb[key]
            if word_emb.has_static_pca:
                return word_emb.pca
        return None

    def update_static_pca(self, key = None):
        if self._word_emb:
            if isinstance(self._word_emb, InputLeaves):
                word_emb = self._word_emb
            else:
                word_emb = self._word_emb[key]
            if word_emb.has_static_pca:
                word_emb.flush_pc_if_emb_is_tuned()

    def state_dict(self, *args, **kwargs):
        odc = super().state_dict(*args, **kwargs)
        emb = self._word_emb
        prefix = '_word_emb.'
        suffix = '_main_emb_layer.weight'
        if isinstance(emb, InputLeaves) and not emb._main_emb_tuned:
            odc.pop(prefix + suffix)
        elif isinstance(emb, nn.ModuleDict):
            for k, v in emb.items():
                if not v._main_emb_tuned:
                    odc.pop(prefix + k + '.' + suffix)
        return odc

    def tensorboard(self, recorder, global_step):
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                params = dict(ContextualRatio = ctx_ratio.mean())
                if ctx_ratio.nelement() > 1:
                    params['RatioStdv'] = ctx_ratio.std()
                recorder.tensorboard(global_step, 'Parameters/%s', **params)

    @property
    def message(self):
        # bsb = self._subject_bias.bias
        # msg = f'BlockySoftmax.bias: {bsb.mean()}'
        # if bsb.nelement() > 1:
        #     msg += f'±{bsb.std()}'
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                ctx_ratio *= 100
                msg = 'Contextual Rate:'
                msg += f' {ctx_ratio.mean():.2f}'
                if ctx_ratio.nelement() > 1:
                    msg += f'±{ctx_ratio.std():.2f}%'
                else:
                    msg += '%'
                return msg

from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value
class OutputLayer(nn.Module):
    def __init__(self,
                 stem_fn,
                 model_dim,
                 num_tags,
                 num_labels,
                 stem_layer,
                 tag_label_layer,
                 **kwargs_forwarding):
        super().__init__(**kwargs_forwarding)

        self._stem_layer = stem_fn(model_dim, **stem_layer)

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
                key = None,
                small_endian_tags = None,
                **kw_args):

        assert isinstance(small_endian_tags, bool)
        (layers_of_base, layers_of_existence,
         stem_specs) = self._stem_layer(bottom_existence,
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
                tag_fn = self._tag_layer if key is None else self._tag_layer[key]
                if small_endian_tags:
                    tags = layers_of_hidden[:, :batch_len]
                else:
                    tags = layers_of_hidden[:, -batch_len:]
                tags = tag_fn(tags)
            
            if self._label_layer is None or ingore_logits:
                labels = None
            else:
                label_fn = self._label_layer if key is None else self._label_layer[key]
                labels = label_fn(layers_of_hidden)
        else:
            layers_of_hidden = tags = labels = None

        return layers_of_base, layers_of_existence, layers_of_hidden, tags, labels, stem_specs
        # return layers_of_base, layers_of_hidden, layers_of_existence, layers_of_orient, tags, labels, trapezoid_info
        # return existence, embeddings, weight, fence, fence_idx, fence_vote, tags, labels, segment, seg_length

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim
        
    @property
    def stem(self):
        return self._stem_layer
        
    def get_label(self, hidden, key = None):
        label_fn = self._label_layer if key is None else self._label_layer[key]
        return label_fn(hidden)

    def get_decision(self, logits):
        return get_decision(self._logit_max, logits)

    def get_decision_with_value(self, logits):
        return get_decision_with_value(self._score_fn, logits)

    def get_multilingual(self, get_label):
        layer = self._label_layer if get_label else self._tag_layer
        assert isinstance(layer, nn.ModuleDict)
        corps = list(layer.keys())
        for eid, lhs in enumerate(corps):
            for rhs in corps[eid:]:
                d = reduce_matrix(layer[lhs].weight, layer[rhs].weight, distance)
                c = reduce_matrix(layer[lhs].weight, layer[rhs].weight, cosine)
                yield lhs, rhs, d, c

    @property
    def message(self):
        messages = []
        if hasattr(stem := self._stem_layer, 'message'):
            messages.append(stem.message)
        if hasattr(self, 'message'):
            messages.append(self.message)
        return '\n'.join(messages)

def distance(lhs, rhs):
    diff = lhs - rhs
    diff = (diff ** 2).sum(2)
    return diff.sqrt() / lhs.shape[1]

def cosine(lhs, rhs):
    lr = (lhs * rhs).sum(2)
    l2 = (lhs * lhs).sum(2)
    r2 = (rhs * rhs).sum(2)
    return lr / (l2 * r2)

def reduce_matrix(lhs, rhs, fn): # [num, dim]
    return fn(lhs.unsqueeze(1).detach(), rhs.unsqueeze(0).detach()).cpu().numpy()


char_rnn_config = dict(embed_dim    = hidden_dim,
                       drop_out     = frac_4,
                       rnn_drop_out = frac_2,
                       module       = rnn_module_type,
                       num_layers   = num_ori_layer,
                       trainable_initials = false_type)
from models.utils import math, init, birnn_fwbw, fencepost, Bias
from models.utils import condense_helper, condense_left
class PadRNN(nn.Module):
    def __init__(self,
                 num_chars,
                 attention_hint, # dims
                 linear_dim, # 01+ fence_vote, activation
                 embed_dim,
                 fence_dim,
                 drop_out,
                 num_layers,
                 module, # num_layers, rnn_drop_out
                 rnn_drop_out,
                 trainable_initials,
                 fence_vote = None,
                 activation = None,
                 char_space_idx = None):
        super().__init__()
        single_size = fence_dim // 2
        if num_layers:
            self._fence_emb = module(embed_dim, single_size,
                                     num_layers    = num_layers,
                                     bidirectional = True,
                                     batch_first   = True,
                                     dropout = rnn_drop_out if num_layers > 1 else 0)
        else:
            self._fence_emb = None
        self._tanh = nn.Tanh()
        bound = 1 / math.sqrt(single_size)
        if trainable_initials:
            c0 = torch.empty(num_layers * 2, 1, single_size)
            h0 = torch.empty(num_layers * 2, 1, single_size)
            self._c0 = nn.Parameter(c0, requires_grad = True)
            self._h0 = nn.Parameter(h0, requires_grad = True)
            init.uniform_(self._c0, -bound, bound)
            init.uniform_(self._h0, -bound, bound)
            self._initial_size = single_size
        else:
            self.register_parameter('_h0', None)
            self.register_parameter('_c0', None)
            self._initial_size = None

        if char_space_idx is None:
            self._pad = nn.Parameter(torch.empty(1, 1, single_size), requires_grad = True)
            init.uniform_(self._pad, -bound, bound)
        else:
            self._pad = char_space_idx
        self._stem_dp = nn.Dropout(drop_out)

        if num_chars: # forward is open
            self._char_emb = nn.Embedding(num_chars, embed_dim, padding_idx = 0)

        if attention_hint: # domain_and_subject is open
            if not isinstance(attention_hint, HParams): attention_hint = HParams(attention_hint)
            self._domain = nn.Linear(fence_dim, embed_dim, bias = False) if attention_hint.get('boundary') else None
            self._subject_unit  = nn.Linear(embed_dim, embed_dim, bias = False) if attention_hint.unit else None
            self._subject_state = nn.Linear(fence_dim, embed_dim, bias = False) if attention_hint.state else None
            single_size = fence_dim // 2
            if attention_hint.before:
                self._subject_fw_b = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_b = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_b = None
                self._subject_bw_b = None
            if attention_hint.after:
                self._subject_fw_a = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_a = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_a = None
                self._subject_bw_a = None
            if attention_hint.difference:
                self._subject_fw_d = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_d = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_d = None
                self._subject_bw_d = None
            # self._subject_bias = Bias(embed_dim) # 0 ~ useless

        if linear_dim:
            if fence_vote is None:
                self._fence_vote = None
                self._fence_l1 = nn.Linear(fence_dim, linear_dim)
                if linear_dim == 1:
                    self._fence_l2 = self._fence_act = lambda x: x
                else:
                    self._fence_act = activation()
                    self._fence_l2 = nn.Linear(linear_dim, 1)
            else:
                self._fence_act = activation()
                from_unit, method = fence_vote.split('.')
                from_unit = from_unit == 'unit'
                if method == 'dot':
                    self._fence_l1 = nn.Linear(fence_dim, linear_dim)
                    if from_unit:
                        self._fence_l2 = nn.Linear(embed_dim, linear_dim)
                    else:
                        self._fence_l2 = nn.Linear(fence_dim, linear_dim)
                    method = self.predict_fence_2d_dot
                elif method == 'cat':
                    if from_unit:
                        self._fence_l1 = nn.Linear(fence_dim + embed_dim, linear_dim)
                    else:
                        self._fence_l1 = nn.Linear(fence_dim << 1, linear_dim)
                    self._fence_l2 = nn.Linear(linear_dim, 1)
                    method = self.predict_fence_2d_cat
                else:
                    raise ValueError('Unknown method: ' + method)
                self._fence_vote = from_unit, method
        # fence_p: f->hidden [b, s+1, h]
        # fence_c: u->hidden [b, s, h]
        # pxc: v->vote [b, s+1, s]
        # fence: s->score [b, s+1] .sum() > 0

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._tanh(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def pad_fwbw_hidden(self, fence_hidden, existence):
        pad = self._stem_dp(self._pad)
        pad = self._tanh(pad)
        return birnn_fwbw(fence_hidden, pad, existence)

    def domain_and_subject(self, fw, bw, fence_idx, unit_emb, fence_hidden):
        if self._domain:
            dom_emb = self._domain(fencepost(fw, bw, fence_idx))
            dom_emb = self._stem_dp(dom_emb)
        else:
            dom_emb = None
        sub_emb = 0 #self._stem_dp(self._subject_bias())
        if self._subject_unit:  sub_emb = sub_emb + self._stem_dp(self._subject_unit(unit_emb))
        if self._subject_state: sub_emb = sub_emb + self._stem_dp(self._subject_state(fence_hidden))
        if self._subject_fw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_a(fw[:, 1:]))
        if self._subject_bw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_a(bw[:, :-1]))
        if self._subject_fw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_b(fw[:, :-1]))
        if self._subject_bw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_b(bw[:, 1:]))
        if self._subject_fw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_d(fw[:, 1:] - fw[:, :-1]))
        if self._subject_bw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_d(bw[:, :-1] - bw[:, 1:]))
        return dom_emb, sub_emb

    def forward(self, char_idx, fence = None, offset = None): # concat fence vectors
        batch_size, char_len = char_idx.shape
        char_emb = self._char_emb(char_idx)
        char_emb = self._stem_dp(char_emb)
        fence_hidden, _ = self._fence_emb(char_emb, self.get_h0c0(batch_size))
        if fence is None:
            helper = condense_helper(char_idx == self._pad, True, offset)
            fence_hidden = fence_hidden.view(batch_size, char_len, 2, -1)
            fw = fence_hidden[:, :, 0]
            bw = fence_hidden[:, :, 1]
        else:    
            existence = char_idx > 0
            fw, bw = birnn_fwbw(fence_hidden, self._tanh(self._pad), existence)
            helper = condense_helper(fence, True, offset)
        fw = condense_left(fw, helper)
        bw = condense_left(bw, helper)
        # select & concat: fw[:*-1] - fw[*1:] & bw...
        return PadRNN.diff_emb(fw, bw)

    @staticmethod
    def diff_emb(fw, bw):
        return torch.cat([fw[:, 1:] - fw[:, :-1], bw[:, :-1] - bw[:, 1:]], dim = 2)

    def predict_fence(self, fw, bw):
        fence = torch.cat([fw, bw], dim = 2)
        fence = self._fence_l1(fence)
        fence = self._stem_dp(fence)
        fence = self._fence_act(fence)
        return self._fence_l2(fence).squeeze(dim = 2)

    def predict_fence_2d_dot(self, fw, bw, hidden, seq_len): # TODO not act for unit
        fence = torch.cat([fw, bw], dim = 2)
        fence = self._fence_l1(fence)
        fence = self._fence_act(self._stem_dp(fence))
        unit = self._fence_l2(hidden)
        unit = self._fence_act(self._stem_dp(unit))
        vote = torch.bmm(fence, unit.transpose(1, 2)) # [b, s+1, s]
        third_dim = torch.arange(unit.shape[1], device = hidden.device)
        third_dim = third_dim[None, None] < seq_len[:, None, None]
        third_dim = torch.where(third_dim, vote, torch.zeros_like(vote))
        return third_dim, third_dim.sum(dim = 2)
    
    def predict_fence_2d_cat(self, fw, bw, hidden, seq_len):
        fence = torch.cat([fw, bw], dim = 2)
        _, fence_len, fence_dim = fence.shape
        batch_size, seg_len, hidden_dim = hidden.shape
        fence = fence[:, :, None].expand(batch_size, fence_len, seg_len, fence_dim)
        hidden = hidden[:, None].expand(batch_size, fence_len, seg_len, hidden_dim)
        vote = torch.cat([fence, hidden], dim = 3) # [b, s+1, s, e]
        vote = self._fence_l1(vote)
        vote = self._stem_dp(vote)
        vote = self._fence_act(vote)
        vote = self._fence_l2(vote).squeeze(dim = 3)
        third_dim = torch.arange(seg_len, device = hidden.device)
        third_dim = third_dim[None, None] < seq_len[:, None, None]
        third_dim = torch.where(third_dim, vote, torch.zeros_like(vote))
        return third_dim, third_dim.sum(dim = 2)