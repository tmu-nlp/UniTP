from models.accp import torch, nn, MultiStem
from models.utils import condense_helper, condense_left, blocky_softmax, blocky_max, bool_start_end
from sys import stderr
from time import time

def predict_disco_hint(dis_batch, bbt_zeros, existence, seq_idx, discontinuous, fence_logits):
    continuous = existence & ~discontinuous
    # 012345678
    # 001132432
    # 10101_11_1 (sup 1_1)
    # 1111001001 -> 101011
    # 1111100100 -> 101011
    # 101011 (min or avg)
    # 111100100
    # 1111100100
    #     ^cz
    # 1111001001
    batch_dim, con_min, con_max = bool_start_end(continuous)
    con_left = torch.cat([bbt_zeros, continuous], dim = 1)
    con_right = torch.cat([continuous, bbt_zeros], dim = 1)
    con_left[batch_dim, con_min] = True
    con_right[batch_dim, con_max + 1] = True
    if True:
        fl_left  = condense_left(fence_logits, condense_helper(con_left,  as_existence = True))
        fl_right = condense_left(fence_logits, condense_helper(con_right, as_existence = True))
        fence_logits = torch.where(fl_left * fl_right > 0, # same sign
                                   torch.where(fl_left > 0,
                                               torch.where(fl_left > fl_right, fl_left, fl_right),
                                               torch.where(fl_left < fl_right, fl_left, fl_right)),
                                   fl_left + fl_right) # oppo sign
    else:
        con_left ^ con_right

    dis_exist = discontinuous[dis_batch]
    dis_helper = condense_helper(dis_exist, as_existence = True)
    dis_indice = condense_left(seq_idx, dis_helper)
    get_dis_emb = lambda emb: condense_left(emb[dis_batch], dis_helper)
    # 001132324 (sup)
    # 012345678
    # 01238 4567 (con_indice & dis_indice)
    # 028   56 (blocky_max: boolean positions; lft|max) | 014   23 (cumsum & take out| concatenated & sorted)
    # 001112334 <- 112223445
    #     3232 (@4567)
    # 001132324 (sup)
    return fence_logits, dis_exist.sum(dim = 1), dis_indice, get_dis_emb

def predict_disco_sections(disco_2d, layer_fence, dis_batch, dis_length, dis_indice):
    dis_size, dis_max_children, _ = disco_2d.shape
    fb_comps = torch.ones(1, dis_max_children, dtype = torch.bool, device = dis_batch.device)
    fb_comp_idx = torch.zeros(dis_max_children, dtype = dis_batch.dtype, device = dis_batch.device)

    # dis_max_parents = dis_max_children >> 1
    # if dis_max_parents > 7:
    #     dis_max_parents = 7
    # dis_max_parents += 1 # make room for comparison [base, comp, comp, |]
    # # TODO - Q: How many components statistically are? Q: density of disco signals (layerwisely)
    # # CharLSTM; VisVote; map_ vmap?
    # start = time()
    # dis_u, dis_d, dis_v = torch.svd_lowrank(disco_2d, dis_max_parents) # [b, mp, mc], [b, mp]
    # dis_d_diff = dis_d[:, :dis_max_parents - 1] - dis_d[:, 1:dis_max_parents] # [b, mp]
    # dis_d_diff_argmax = dis_d_diff.argmax(dim = 1, keepdim = True)
    # seq_idx = torch.arange(dis_max_parents, device = dis_batch.device)
    # dis_d[dis_d_diff_argmax < seq_idx[None]] = 0
    # clear_3d = torch.bmm(dis_d.unsqueeze(dim = 1) * dis_u, dis_v.transpose(1, 2))
    # svd_time = time() - start
    seq = torch.arange(dis_max_children, device = dis_batch.device)
    clear_3d = disco_2d
    clear_3d[:, seq, seq] = 1

    dis_comps = []
    for clear_2d, dis_len, bid, indice in zip(clear_3d, dis_length, dis_batch, dis_indice):
        #                                       final      first     final      first
        # 4578
        # 1010 (7); 0101 (5) | *****1*1*
        # (32) -> 3030; 0202 -> 3232
        # 001112334 -> ****3232*
        threshold = 0.5
        clear_2d = clear_2d[:dis_len, :dis_len] # no way: need var sizes
        b_clear_2d = clear_2d > threshold # clear_2d.mean() # mean ? softer? bad for ones; CPU or GPU?  inclusion matrix for comp
        in_deg  = b_clear_2d.sum(dim = 0)
        out_deg = b_clear_2d.sum(dim = 1)
        # import pdb; pdb.set_trace()
        if (in_deg != out_deg).any():
            comps = fb_comps[:, :dis_len]
            comp_idx = fb_comp_idx[:dis_len]
        else:
            comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True) # no way: map_fn
            while not (comps.sum(dim = 0) == 1).all():
                if (clear_2d > threshold).any():
                    threshold = clear_2d[clear_2d > threshold].min()
                    b_clear_2d = clear_2d > threshold
                    comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True)
                else:
                    threshold = None
                    break
            if threshold is None:
                # import pdb; pdb.set_trace()
                comps = fb_comps[:, :dis_len]
                comp_idx = fb_comp_idx[:dis_len]

        # num_comps, _ = comps.shape
        # comp_idx = (comp_dim[:num_comps, None] * comps).sum(dim = 0) # 0101 (or 1010)
        dis_b_max = blocky_max(comp_idx, clear_2d.sum(dim = 0), False) # 0110 (for 57)
        comp_max_idx = comp_idx[dis_b_max] # 0101 -> 01 or 10
        
        indice = indice[:dis_len]
        dis_max_idx = indice[dis_b_max] # 4578 -> 57
        layer_fence[bid, dis_max_idx] = True # 00101_11_1 -> 
        dis_comps.append((bid, dis_max_idx, comps, comp_max_idx, indice))
        # print(dis_max_idx)
    
    # dis_comps = [None] * dis_size
    # def disco_inject(dis_bid, con_bid, dis_len):
    #     clear_2d = clear_3d[dis_bid, :dis_len, :dis_len]
    #     b_clear_2d = clear_2d > 0.5
    #     in_deg  = b_clear_2d.sum(dim = 0)
    #     out_deg = b_clear_2d.sum(dim = 1)
    #     if (in_deg != out_deg).any() or (in_deg <= 1).any() or (out_deg <= 1).any():
    #         comps = fb_comps[:, :dis_len]
    #         comp_idx = fb_comp_idx[:dis_len]
    #     else:
    #         comps, comp_idx = b_clear_2d.unique(dim = 0, return_inverse = True) # no way: map_fn
    #         if not (comps.sum(dim = 0) == 1).all():
    #             comps = fb_comps[:, :dis_len]
    #             comp_idx = fb_comp_idx[:dis_len]

    #     dis_b_max = blocky_max(comp_idx, clear_2d.sum(dim = 0), False)
    #     comp_max_idx = comp_idx[dis_b_max] # 0101 -> 01 or 10
    #     dis_max_idx = dis_indice[dis_bid, :dis_len][dis_b_max]
    #     layer_fence[con_bid, dis_max_idx] = True # 00101_11_1 -> 
    #     dis_comps[dis_bid] = dis_max_idx, comps, comp_max_idx
    #     return dis_bid

    # dis_idx = torch.arange(dis_size, device = dis_batch.device)
    # dis_idx.map2_(dis_batch, dis_length, disco_inject)

    # print(layer_fence * 1)
    sections = layer_fence.cumsum(dim = 1) # 0011123444
    # print(sections)
    # print('sec')

    # def disco_assign(dis_bid, con_bid, dis_len):
    #     indice = dis_indice[dis_bid, :dis_len]
    #     dis_max_idx, comps, comp_max_idx = dis_comps[dis_bid]
    #     order = sections[con_bid, dis_max_idx][comp_max_idx]
    #     # print(con_bid, sections[con_bid, dis_max_idx])
    #     # print(comp_max_idx, sections[con_bid, dis_max_idx])
    #     # print((order[:, None] * comps).sum(dim = 0))
    #     sections[con_bid, indice] = (order[:, None] * comps).sum(dim = 0) # 3232 for 0101
    #     # print()
    #     return dis_bid
    # dis_idx.map2_(dis_batch, dis_length, disco_assign)

    for bid, dis_max_idx, comps, comp_max_idx, indice in dis_comps:
        # 0:0101  23 2
        # 1:1010  01 3 3232
        # 0:1010     3
        # 1:0101  10 2 3232
        order = sections[bid, dis_max_idx][comp_max_idx]
        sections[bid, indice] = (order[:, None] * comps).sum(dim = 0) # 3232 for 0101
    # cate_time = time() - start - svd_time
    # with open('svd_time.csv', 'a+') as fw:
    #     fw.write(f'{dis_size},{dis_max_children},{svd_time},{cate_time}\n')
    return sections


from models.types import rnn_module_type, discontinuous_attention_hint, activation_type, logit_type, fmin, fmax
from models.combine import get_combinator, get_components, valid_trans_compound
from utils.types import orient_dim, hidden_dim, frac_2, frac_4, frac_5, num_ori_layer, true_type, false_type, BaseType
disco_2d_from = BaseType(2, as_index = True, default_set = ('state.dot', 'unit.dot', 'state_unit.dot', 'unit_state.dot'
                                                            'state.cat', 'unit.cat', 'state_unit.cat', 'unit_state.cat'))
stem_config = dict(space_dim           = orient_dim,
                #    disco_dim           = orient_dim,
                   disco_indie_io      = false_type,
                   disco_1d_activation = activation_type,
                   disco_2d_activation = activation_type,
                   disco_2d_from       = disco_2d_from,
                   disco_linear_dim    = orient_dim,
                   fence_linear_dim    = orient_dim,
                   space_module        = rnn_module_type,
                   fence_activation    = activation_type,
                   attention_hint      = discontinuous_attention_hint,
                   num_layers          = num_ori_layer,
                   drop_out            = frac_4,
                   rnn_drop_out        = frac_2,
                   trainable_initials  = true_type)

class DiscoMultiStem(MultiStem):
    def __init__(self,
                 model_dim,
                 space_dim,
                #  disco_dim,
                 disco_indie_io,
                 disco_1d_activation,
                 disco_2d_activation,
                 disco_2d_from,
                 space_module,
                 disco_linear_dim,
                 fence_linear_dim,
                 fence_activation,
                 attention_hint,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 trainable_initials):
        super().__init__(model_dim,
                         space_dim,
                         fence_linear_dim,
                         space_module,
                         None,
                         fence_activation,
                         attention_hint,
                         num_layers,
                         drop_out,
                         rnn_drop_out,
                         trainable_initials)
        self._disco_1d_l1 = nn.Linear(space_dim, disco_linear_dim)
        self._disco_1d_act = disco_1d_activation()
        self._disco_2d_act = disco_2d_activation()
        self._disco_1d_l2 = nn.Linear(disco_linear_dim, 1)
        disco_2d_from, method = disco_2d_from.split('.')
        if disco_2d_from == 'state':
            self._select = lambda g, s, u: (g(s),) * 2
        elif disco_2d_from == 'unit':
            self._select = lambda g, s, u: (g(u),) * 2
        else:
            self._select = lambda g, s, u: (g(s), g(u))
        if method == 'dot':
            if disco_2d_from == 'state':
                in_dim = out_dim = space_dim
            elif disco_2d_from == 'unit':
                in_dim = out_dim = model_dim
            else:
                in_dim = space_dim
                out_dim = model_dim
            self._disco_2d_i = d2di = nn.Linear(in_dim, disco_linear_dim, bias = disco_indie_io)
            self._disco_2d_o = nn.Linear(out_dim, disco_linear_dim) if disco_indie_io else d2di
            self.predict_2d_disco = self.predict_2d_disco_dot
        else:
            if disco_2d_from == 'state':
                in_dim = space_dim << 1
            elif disco_2d_from == 'unit':
                in_dim = model_dim << 1
            else:
                in_dim = space_dim + model_dim
            self._disco_2d_i = nn.Linear(in_dim, disco_linear_dim, bias = disco_indie_io)
            self._disco_2d_o = nn.Linear(disco_linear_dim, 1)
            self.predict_2d_disco = self.predict_2d_disco_cat

    def predict_1d_disco(self, fence_hidden):
        hidden = self._disco_1d_l1(fence_hidden)
        hidden = self._stem_dp(hidden)
        hidden = self._disco_1d_act(hidden)
        return self._disco_1d_l2(hidden).squeeze(dim = 2)

    def predict_2d_disco_dot(self, in_hidden, out_hidden):
        hidden_i = self._disco_2d_i(in_hidden) # [b, s, e]
        hidden_o = self._disco_2d_o(out_hidden)
        hidden_i = self._stem_dp(hidden_i)
        hidden_o = self._stem_dp(hidden_o).transpose(1, 2) # [b, e, s]
        hidden_i = self._disco_2d_act(hidden_i)
        hidden_o = self._disco_2d_act(hidden_o)
        return torch.matmul(hidden_i, hidden_o) # [b, s, s]

    def predict_2d_disco_cat(self, in_hidden, out_hidden):
        batch_size, seq_dim, in_dim = in_hidden.shape
        _, _, out_dim = out_hidden.shape
        in_hidden = in_hidden[:, :, None].expand(batch_size, seq_dim, seq_dim, in_dim)
        out_hidden = out_hidden[:, None].expand(batch_size, seq_dim, seq_dim, out_dim)
        hidden = torch.cat([in_hidden, out_hidden], dim = 3)
        hidden = self._disco_2d_i(hidden)
        hidden = self._stem_dp(hidden)
        hidden = self._disco_2d_act(hidden)
        return self._disco_2d_o(hidden).squeeze(dim = 3)

    def forward(self, unit_emb, existence, supervision = None, **kw_args):
        batch_size, seg_len, model_dim = unit_emb.shape
        h0c0 = self.get_h0c0(batch_size)
        max_iter_n = seg_len << 2 # 4 times
        teacher_forcing =  supervision is not None
        segment, seg_length = [], []
        batch_dim = torch.arange(batch_size, device = unit_emb.device)
        bbt_zeros = torch.zeros(batch_size, 1, dtype = torch.bool, device = existence.device)
        bbt_ones  = torch.ones (batch_size, 1, dtype = torch.bool, device = existence.device)

        layers_of_u_emb = []
        layers_of_existence = []
        # training logits
        layers_of_fence = [] # 101011
        layers_of_disco_1d = [] # 0000111100
        layers_of_disco_2d = [] # 1010x0101
        
        if teacher_forcing:
            space, dis_disco = supervision
        else:
            # dis_slice_start = 0
            layers_of_space = [] # 001132324
            # dbg/vis signal
            # layers_of_slice = []
            # layers_of_shape = [] # 
            layers_of_weight = []

        for l_cnt in range(max_iter_n):
            seq_len = existence.sum(dim = 1)
            layers_of_u_emb.append(unit_emb)
            layers_of_existence.append(existence)
            if not teacher_forcing:
                segment   .append(seg_len)
                seg_length.append(seq_len)

            disco_hidden, _ = self._fence_emb(unit_emb, h0c0)
            fw, bw = self.pad_fwbw_hidden(disco_hidden, existence)
            disco_hidden = self._stem_dp(disco_hidden) # the order should be kept
            fw = self._stem_dp(fw)
            bw = self._stem_dp(bw)
            disco_1d_logits = self.predict_1d_disco(disco_hidden)
            layers_of_disco_1d.append(disco_1d_logits)
            fence_logits = self.predict_fence(fw, bw) # local fw & bw for continuous
            layer_fence_logits = fence_logits # save for the next function

            longer_seq_idx = torch.arange(seg_len + 1, device = unit_emb.device)
            seq_idx = longer_seq_idx[None, :seg_len]
            if teacher_forcing:
                discontinuous = dis_disco[l_cnt] # [b, s]
            else:
                discontinuous = disco_1d_logits > 0
                if discontinuous.shape != existence.shape:
                    print(f'WARNING: Invalid sections caused unmatched existence {l_cnt}', file = stderr, end = '')
                    break
                discontinuous &= existence
                discontinuous = torch.where(discontinuous.sum(dim = 1, keepdim = True) == 1,
                                            torch.zeros_like(discontinuous), discontinuous)
            
            if seg_len == 1:
                break # teacher forcing or a good model
            elif len(seg_length) > 1:
                prev, curr = seg_length[-2:]
                if (prev == curr).all():
                    break
                elif l_cnt == max_iter_n - 1:
                    print(f'WARNING: Action layers overflow maximun {l_cnt}', file = stderr, end = '')
                    break
                
            dis_batch = discontinuous.any(dim = 1)
            if dis_batch.any():
                dis_batch, = torch.where(dis_batch)
                dis_batch_size, = dis_batch.shape
                (fence_logits, dis_length, dis_indice,
                 get_dis_emb) = predict_disco_hint(dis_batch, bbt_zeros,
                                                   existence, seq_idx.repeat(dis_batch_size, 1),
                                                   discontinuous, fence_logits)
                disco_2d_logits = self.predict_2d_disco(*self._select(get_dis_emb, disco_hidden, unit_emb))
            else:
                disco_2d_logits = None
            layers_of_fence.append(fence_logits)

            if teacher_forcing:
                sections = space[l_cnt]
                if disco_2d_logits is not None:
                    layers_of_disco_2d.append(disco_2d_logits.reshape(-1))
            else:
                layer_len = seq_len[:, None]
                layer_fence_logits[:, 0] = fmax
                layer_fence_logits[batch_dim, seq_len] = fmax
                layer_fence_logits[longer_seq_idx > layer_len] = fmin
                layer_fence = (layer_fence_logits > 0)[:, :-1]

                if disco_2d_logits is None:
                    sections = layer_fence.cumsum(dim = 1)
                    # layers_of_slice.append((dis_slice_start, dis_slice_start))
                    # layers_of_shape.append(None)
                else:
                    # layers_of_space.append(disco_2d.shape)
                    # dis_slice_end = dis_slice_start + disco_2d.shape.numel()
                    # layers_of_slice.append((dis_slice_start, dis_slice_end))
                    # 012345678
                    # 000011011 (disc.)
                    #_0101?11?1 v
                    #0010__1__1 v (result: layer_fence)
                    #000011011_ ^
                    layer_fence &= discontinuous.logical_not()
                    disco_2d = self._sigmoid(disco_2d_logits)
                    layers_of_disco_2d.append(disco_2d.reshape(-1))
                    sections = predict_disco_sections(disco_2d, layer_fence, dis_batch, dis_length, dis_indice)
                sections = torch.where(seq_idx < layer_len, sections, torch.zeros_like(sections))
                layers_of_space.append(sections)
                
            sub_emb = self._stem_dp(self._subject_bias())
            if self._subject_unit:  sub_emb = sub_emb + self._stem_dp(self._subject_unit(unit_emb))
            if self._subject_state: sub_emb = sub_emb + self._stem_dp(self._subject_state(disco_hidden))
            if self._subject_fw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_a(fw[:, 1:]))
            if self._subject_bw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_a(bw[:, :-1]))
            if self._subject_fw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_b(fw[:, :-1]))
            if self._subject_bw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_b(bw[:, 1:]))
            if self._subject_fw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_d(fw[:, 1:] - fw[:, :-1]))
            if self._subject_bw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_d(bw[:, :-1] - bw[:, 1:]))
            weights, unit_emb = blocky_softmax(sections, sub_emb, None, unit_emb)

            seg_len   = unit_emb.shape[1]
            existence, _ = sections.max(dim = 1, keepdim = True)
            existence = seq_idx[:, :seg_len] < existence
            if not teacher_forcing:
                layers_of_weight.append(weights)

        embeddings = torch.cat(layers_of_u_emb,     dim = 1)
        existence  = torch.cat(layers_of_existence, dim = 1)
        fence      = torch.cat(layers_of_fence,     dim = 1)
        disco_1d   = torch.cat(layers_of_disco_1d, dim = 1)
        disco_2d   = torch.cat(layers_of_disco_2d, dim = 0) if layers_of_disco_2d else None
        if teacher_forcing:
            weight = space = None
        else:
            if not layers_of_space:
                assert not layers_of_weight
                space  = torch.zeros(batch_size, 0, dtype = batch_dim.dtype, device = batch_dim.device)
                weight = torch.zeros(batch_size, 0, model_dim, dtype = unit_emb.dtype, device = unit_emb.device)
            else:
                space  = torch.cat(layers_of_space,  dim = 1)
                weight = torch.cat(layers_of_weight, dim = 1)
            seg_length = torch.stack(seg_length, dim = 1)

        return existence, embeddings, weight, disco_1d, fence, disco_2d, space, segment, seg_length


# batch_insert(101011, 4444) -> 1010000011
#  001132324    001112334 [_-]=[01]
# 1010000011 -> _010_--_1:1 -> ****3232*
# or x inter
# batch_insert(_01011, 4444, 0110) -> 0010011011
# 001112334 001132324


# 000011110  4444
# 000011011  4455
# 4567 diff-> 111 -> 1000 -> 1111 -> 4444
# 4578 diff-> 121 -> 1010 -> 1122 -> 4455


multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

model_type = dict(space_layer     = stem_config,
                  tag_label_layer = multi_class)
from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value, get_loss

class BaseRnnTree(DiscoMultiStem):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 space_layer,
                 tag_label_layer,
                 **kw_args):
        super().__init__(model_dim, **space_layer)

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
        (existence, embeddings, weight, disco_1d, fence, disco_2d, space, segment,
         seg_length) = super().forward(base_inputs, bottom_existence, **kw_args)

        if self._hidden_dim:
            layers_of_hidden = self._shared_layer(embeddings)
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

        return existence, embeddings, weight, disco_1d, fence, disco_2d, space, tags, labels, segment, seg_length

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

    def get_losses(self, batch, tag_logits, label_logits):
        height_mask = batch['segment'][None] * (batch['seg_length'] > 0)
        height_mask = height_mask.sum(dim = 1)
        tag_loss   = get_loss(self._tag_layer,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(self._label_layer, self._logit_max, label_logits, batch, False, height_mask, None, 'label')
        return tag_loss, label_loss