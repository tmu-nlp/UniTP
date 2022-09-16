import torch
from data.continuous import Signal
from data.continuous.binary import X_RGT
from data.dataset import LengthOrderedDataset, np
from data.dataset import pad_tag_like_nil, pad_tag_like_bos_eos
from data.dataset import pad_label_like_nil, pad_label_like_bos_eos
from data.dataset import erect_joint_like, erect_split_like
from utils.types import F_CNF, F_CON, F_RANDOM
from collections import namedtuple

NewBin = namedtuple('NewBin', 'sentence_level, left, right, msub')
OldBin = namedtuple('OldBin', 'cache, right')

from data.mp import Process, Rush, Pool

class WorkerX(Process):
    S = 'S'

    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        i, q, fileids, reader, from_corpus, serialize_esub = self._args
        error_cnt = inst_cnt = n_proceed = l_estimate = 0
        n_fileid = len(fileids)

        for eid, fn in enumerate(fileids):
            trees = reader.parsed_sents(fn)
            n_proceed += len(trees)
            n_estimate = int(n_proceed * n_fileid / (eid + 1))

            for tree in trees:
                q.put((i, n_estimate) if n_estimate != l_estimate else i)
                try:
                    dx = from_corpus(tree)
                    dx.binary(F_RANDOM)
                    dx.multib()
                except:
                    error_cnt += 1
                    continue
            
                q.put((WorkerX.S, dx.serialize(serialize_esub)))
                inst_cnt += 1
        q.put((i, inst_cnt, error_cnt))


class ContinuousDataset(LengthOrderedDataset):
    __D_ESB = 1 << 0
    __D_RGT = 1 << 1

    def __init__(self,
                 binary,
                 reader,
                 fileids,
                 from_fn,
                 v2is,
                 char_as_token,
                 paddings,
                 factor,
                 esub,
                 msub,
                 b_condense_or_m_joint,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None,
                 self_check_i2vs = None):

        error_count   = 0
        w2i, t2i, l2i = v2is
        length, token, tag, signals, text = [], [], [], [], []
        rush = Rush(WorkerX, fileids, reader, from_fn, esub > 0)
        def receive(t, qbar):
            if isinstance(t, int):
                qbar.update(t)
            elif len(t) == 2:
                tid, n_estimate = t
                if isinstance(tid, int):
                    qbar.update(tid, total = n_estimate)
                    qbar.update(tid)
                else:
                    signal = Signal.instantiate(t[1])
                    length .append(signal.max_height)
                    token  .append(signal.char_to_idx(w2i) if char_as_token else signal.word_to_idx(w2i))
                    tag    .append(signal. tag_to_idx(t2i))
                    text   .append(signal.word)
                    signals.append(signal)
            else:
                nonlocal error_count
                i, tc, ec = t
                error_count += ec
                return i, tc
        rush.mp_while(False, receive)
        if error_count:
            print(error_count, 'error(s)')

        heads = 'tree', 'token'
        if factor is None:
            label = extra = None
        else:
            if binary:
                _heads = 'tag', 'label', 'xtype'
                factor, extra = self.reset_binary_factor(factor, esub, msub, initialize = len(length))
            else:
                _heads = 'tag', 'label', 'chunk'
                factor = self.reset_multib_factor(esub, msub, initialize = True)
                if msub == 0:
                    extra = tuple([None, None] for _ in range(len(length)))
                else:
                    extra = msub
            heads = heads + _heads
            label = 'label'
        if char_as_token:
            heads = heads + ('char_segment',)

        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, w2i)

        super().__init__(heads, label, length, factor, min_len, max_len, extra_text_helper)
        self._args = token, tag, signals, l2i, char_as_token, paddings, binary, b_condense_or_m_joint, self_check_i2vs, extra

    def reset_binary_factor(self, factor, esub, msub, *, initialize = None):
        level, left, right = factor
        esub_factor = factor = {0: 1 - esub, 1: esub}
        sentence_level = level == 'sentence'
        # sentence-cnf-num: old a little dynamic; sentence/phrase-continuous-num: old static
        # phrase-cnf-num or old but msub > 0; sentence/phrase-num-num: beta(num, num)
        if old_cnf := (sentence_level and left in (F_CNF, F_CON) and msub == 0):
            if initialize:
                cache = tuple([None, None, None, None] for _ in range(initialize))
            if left == F_CNF:
                factor = {} # overwrite
                for k, v in esub_factor.items():
                    factor[k * ContinuousDataset.__D_ESB] = (1 - right) * v # left
                    factor[k * ContinuousDataset.__D_ESB | ContinuousDataset.__D_RGT] = right * v # right
                extra = cache
            else:
                extra = lambda cache: OldBin(cache, right)
        else:
            extra = NewBin(sentence_level, left, right, msub)
        if initialize:
            return factor, (extra(cache) if callable(extra) else extra)

        if old_cnf:
            if isinstance(old_extra := self._args[-1], NewBin):
                cache = tuple([None, None, None, None] for _ in range(len(self._args[0])))
            else: # reuse old cache
                cache = old_extra[1] if isinstance(old_extra, OldBin) else old_extra
            new_extra = extra(cache) if callable(extra) else cache
        else:
            new_extra = extra
        self._args = self._args[:-1] + (new_extra,)
        super()._reset_factors(factor)

    def reset_multib_factor(self, esub, msub, *, initialize = False):
        esub = {0: 1 - esub, 1: esub}
        if initialize:
            return esub
        super()._reset_factors(esub)
        return msub

    def at_idx(self, idx, factor, helper_outputs):
        token, tag, signals, l2i, char_as_token, _, binary, b_condense_or_m_joint, _, extra = self._args
        signal = signals[idx]
        sample = [signal.tree, token[idx]]
        if extra is not None and not isinstance(extra, bool):
            sample.append(tag[idx])
            if binary:
                if isinstance(extra, NewBin):
                    sentence_level, left, right, msub = extra
                    if isinstance(left, float):
                        continuous = np.random.beta(left, right) if sentence_level else (left, right)
                    elif left == F_CNF:
                        continuous = (np.random.random() < right) if sentence_level else (F_CNF, right)
                    else:
                        assert left == F_CON and sentence_level and msub > 0
                        continuous = right
                    lb, sp = signal.binary(continuous, factor, msub, b_condense_or_m_joint, l2i = l2i)
                else:
                    if static := isinstance(extra, OldBin):
                        cache, cnf_right = extra
                        cache = cache[idx]
                    else:
                        cache = extra[idx]
                    if cache[factor] is None:
                        cnf = cnf_right if static else float(bool(factor & ContinuousDataset.__D_RGT))
                        esb = bool(factor & ContinuousDataset.__D_ESB)
                        cache[factor] = signal.binary(cnf, esb, every_n = b_condense_or_m_joint, l2i = l2i)
                    lb, sp = cache[factor]
            else:
                esb = bool(factor & ContinuousDataset.__D_ESB)
                if isinstance(extra, tuple):
                    cache = extra[idx]
                    if cache[factor] is None:
                        cache[factor] = signal.multib(esb, l2i = l2i, joint = b_condense_or_m_joint)
                    lb, sp = cache[factor]
                else:
                    lb, sp = signal.multib(esb, extra, l2i = l2i, joint = b_condense_or_m_joint) # more_sub
            sample.append(lb)
            sample.append(sp)

        if char_as_token:
            assert not binary
            sample.append(signal.char_segment(joint = b_condense_or_m_joint))
        return tuple(sample)


    def _collate_fn(self, batch, length, segment):
        _, _, _, _, char_as_token, paddings, binary, b_condense_or_m_joint, self_check_i2vs, extra = self._args
        indice_args = dict(device = self.device, dtype = torch.long)
        field_columns = dict(length = length)
        if char_as_token:
            max_token_len = 0
            batch_char_segment = []
            for chars in batch.token:
                if (clen:= len(chars)) > max_token_len:
                    max_token_len = clen
                batch_char_segment.append([clen])
            if b_condense_or_m_joint:
                char_joint = erect_joint_like(batch_char_segment, batch.char_segment, max_token_len, 0)
            else:
                char_split = erect_split_like(batch.char_segment, [max_token_len], 0)
            field_columns['tag_layer'] = 1
        else:
            max_token_len = length.max()

        if paddings:
            max_token_len += 2 # BOS and EOS
            offset = (max_token_len - length) // 2
            field_columns['offset'] = torch.as_tensor(offset, **indice_args)
            field_columns['token'] = torch.as_tensor(pad_tag_like_bos_eos(batch.token, max_token_len, offset, *paddings['token']), **indice_args)
        else:
            field_columns['token'] = torch.as_tensor(pad_tag_like_nil(batch.token, max_token_len), **indice_args)
        if binary:
            field_columns['condense_per'] = b_condense_or_m_joint # trigger traperzoid/triangle

        if extra is None:
            field_columns['tree'] = batch.tree
            if char_as_token:
                if b_condense_or_m_joint: # joint
                    sig = torch.ones(len(length), max_token_len + 1, dtype = torch.bool, device = self.device)
                    idx = torch.as_tensor(char_joint, **indice_args)
                    sig[idx[:, 0], idx[:, 1]] = False
                else: # split (load slower)
                    sig = torch.zeros(len(length), max_token_len + 1, dtype = torch.bool, device = self.device)
                    idx = torch.as_tensor(char_split, **indice_args)
                    sig[idx[:, 0], idx[:, 1]] = True
                field_columns['char_chunk'] = sig
        else:
            max_tag_len = length.max()
            if paddings:
                field_columns['tag'] = torch.as_tensor(pad_tag_like_bos_eos(batch.tag, max_tag_len, offset, *paddings['tag']), **indice_args)
            else:
                field_columns['tag'] = torch.as_tensor(pad_tag_like_nil(batch.tag, max_tag_len), **indice_args)

            batch_segment = segment.max(0)
            if paddings:
                field_columns['label'] = torch.as_tensor(pad_label_like_bos_eos(batch.label, batch_segment, offset, *paddings['label']), **indice_args)
            else:
                field_columns['label'] = torch.as_tensor(pad_label_like_nil(batch.label, batch_segment), **indice_args)

            if binary:
                if paddings:
                    field_columns['xtype'] = torch.as_tensor(pad_label_like_bos_eos(batch.xtype, batch_segment, offset, X_RGT, 0, X_RGT), dtype = torch.uint8, device = self.device)
                else:
                    field_columns['xtype'] = torch.as_tensor(pad_label_like_nil(batch.xtype, batch_segment), dtype = torch.uint8, device = self.device)
                if self_check_i2vs:
                    a_args = self_check_i2vs
                    b_args = [np.zeros_like(length), length]
                    b_args += [field_columns[x].numpy() for x in 'token tag label'.split()]
                    b_args.append((field_columns['xtype'].numpy() & X_RGT) > 0)
                    if b_condense_or_m_joint:
                        from data.continuous.binary.trapezoid import data_to_tree
                        a_args = (batch_segment,) + a_args
                        b_args.append(segment)
                    else:
                        from data.continuous.binary.triangle import data_to_tree
                    for eid, (otree, args) in enumerate(zip(batch.tree, zip(*b_args))):
                        ttree = data_to_tree(*args, *a_args)
                        if otree != ttree:
                            print(eid)
                            print(otree)
                            print(ttree)
                            breakpoint()
            else:
                assert not paddings, 'CM does not cover paddings.'
                sig_size = (batch_segment + 1).sum()
                if char_as_token:
                    sig_size  += max_token_len + 1
                    sig_offset = max_token_len + 1
                    field_columns['tag_layer'] = 1
                else:
                    char_joint = char_split = []
                    sig_offset = 0

                if b_condense_or_m_joint: # joint (load faster)
                    sig = torch.ones(len(length), sig_size, dtype = torch.bool, device = self.device)
                    if joint := (char_joint + erect_joint_like(segment, batch.chunk, batch_segment, sig_offset)):
                        idx = torch.as_tensor(joint, **indice_args)
                        sig[idx[:, 0], idx[:, 1]] = False
                else: # split (load slower)
                    split = char_split + erect_split_like(batch.chunk, batch_segment, sig_offset)
                    sig = torch.zeros(len(length), sig_size, dtype = torch.bool, device = self.device)
                    idx = torch.as_tensor(split, **indice_args)
                    sig[idx[:, 0], idx[:, 1]] = True
                field_columns['chunk'] = sig
                
                if char_as_token:
                    segment = np.concatenate([batch_char_segment, segment], axis = 1)
                    batch_segment = np.concatenate([[max_token_len], batch_segment])
                if self_check_i2vs:
                    from data.continuous.multib.mp import tensor_to_tree
                    a_args = self_check_i2vs +  (int(char_as_token), batch_segment,) 
                    b_args = [field_columns[x].numpy() for x in 'token tag label chunk'.split()]
                    b_args.append(segment)
                    trees = tuple(tensor_to_tree(*a_args, *args) for args in zip(*b_args))
                    for t0, t1 in zip(batch.tree, trees):
                        if t0 != t1:
                            print(t0)
                            print(t1)
                            breakpoint()
                    
            if not binary or b_condense_or_m_joint is not None:
                field_columns['segment'] = segment
                field_columns['batch_segment'] = batch_segment
        return field_columns

        # if field_label := field in ('label', 'polar'):
        #     mask_length = np.zeros([batch_size], dtype)
        #     seg_length = np.zeros([batch_size, max_len], dtype)
        #     top3_label = np.stack([np.concatenate(x[-1:-3:-1]) for x in column]) # [batch, 3]

        # for l_, layer in enumerate(zip_longest(*column)):
        #     max_layer_len = max(len(x) for x in layer if x is not None)
        #     if paddings:
        #         max_layer_len += 2
        #     cumu_length += max_layer_len
        #     l_start = full_triangular_len - cumu_length
        #     l_end   = l_start + max_layer_len
        #     if field_label:
        #         segments.append(max_layer_len)
        #     for i_, seq in enumerate(layer):
        #         if seq is None:
        #             continue
        #         seq_len = len(seq)
        #         if field_label:
        #             mask_length[i_] += max_layer_len
        #             seg_length[i_, -1 - l_] = seq_len
        #         if paddings:
        #             bid, eid = paddings[field]
        #             start = l_start + offsets[i_]
        #             end   = start + seq_len
        #             tensor[i_, l_start:start] = bid
        #             tensor[i_, start:end] = seq
        #             tensor[i_, end:l_end] = eid
        #         else:
        #             end = l_start + seq_len
        #             tensor[i_, l_start:end] = seq
        # tensor = tensor[:, -cumu_length:]

        # field_columns[field] = tensor

        # field_columns['mask_length'] = cumu_length - mask_length
        # field_columns['top3_label']  = top3_label

# def write_tensors(labels, xtypes, tensor_labels, tensor_xtypes, offset, paddings = None, vocab = None, skip_top = 0):
#     tensor_vlen = tensor_labels.shape[0] + skip_top
#     tensor_height, oset = t_index(tensor_vlen)
#     assert oset == 0
#     # assert tensor_labels.shape == tensor_xtypes.shape
#     py_len = len(labels)
#     py_height, oset = t_index(py_len)
#     assert oset == 0
#     assert py_len == len(xtypes)
#     height_diff = tensor_height - py_height
#     assert height_diff >= 0
#     if paddings:
#         l_bos, l_eos, x_bos, x_eos = paddings
#         eos_d = height_diff - offset

#     for src, (lbl, xty) in enumerate(zip(labels, xtypes)):
#         if xty:
#             lid, oset = t_index(src)
#             dst = s_index(lid + height_diff, oset + offset) - skip_top
#             if vocab is not None:
#                 lbl = vocab[lbl]
#             tensor_labels[dst] = lbl
#             tensor_xtypes[dst] = xty
#             if paddings:
#                 if oset == 0:
#                     start = dst - offset
#                     tensor_labels[start:dst] = l_bos
#                     tensor_xtypes[start:dst] = x_bos
#                 if oset == lid:
#                     start = dst + 1
#                     end = start + eos_d
#                     tensor_labels[start:end] = l_eos
#                     tensor_xtypes[start:end] = x_eos


# from data.delta import E_XDIM

# fields = 'token', 'tag', 'ftag'
# fieldx = 'label', 'xtype'
# # FieldOrder = 'token', 'tag', 'label', 'xtype', 'ftag', 'length'

# class PennTreeKeeper:
#     def __init__(self, tree, v2is, trapezoid_height):
#         self._tree = tree
#         self._v2is = v2is
#         self._w_p = None
#         self._factored = {}
#         self._trapezoid_height = trapezoid_height

#     def update_factored(self, factored, words):
#         self._factored.update(factored)
#         tree = self._tree
#         for i, word in enumerate(words):
#             if word == '(':
#                 tree[tree.leaf_treeposition(i)] = '('
#             elif word == ')':
#                 tree[tree.leaf_treeposition(i)] = ')'

#     def __getitem__(self, factor):
#         if factor in self._factored:
#             return self._factored[factor]

#         w2i, t2i, l2i, x2i = self._v2is
#         dx, _ = TreeKeeper.from_penn(self._tree, factor, do_preproc = False) # [not here] watch for keyaki arg wordtrace for adjust_label
#         if self._w_p is None:
#             word, tag = dx.word_tag(w2i, t2i)
#             word = np.asarray(word)
#             tag  = np.asarray(tag)
#             self._w_p = word, tag
#         else:
#             word, tag = self._w_p

#         layers_of_labels = []
#         layers_of_xtypes = []
#         for labels, xtypes in dx.trapezoid_gen(self._trapezoid_height, l2i, x2i):
#             labels = np.asarray(labels)
#             xtypes = np.asarray(xtypes)
#             layers_of_labels.append(labels)
#             layers_of_xtypes.append(xtypes)

#         factored = dict(token = word,
#                         tag   = tag,
#                         label = layers_of_labels,
#                         xtype = layers_of_xtypes)
#         self._factored[factor] = factored
#         return factored

#     def __str__(self):
#         s = f'Keeper with ' + ', '.join(self._factored.keys()) + 'cached'
#         return s

# from unidecode import unidecode
# class StanTreeKeeper:
#     def __init__(self, line, v2is, trapezoid_height):
#         self._line = line
#         self._v2is = v2is
#         self._factored = None
#         self._trapezoid_height = trapezoid_height

#     def update_factored(self, factored):
#         self._factored = factored

#     def get(self):
#         if self._factored is None:

#             w2i, p2i, x2i = self._v2is
#             tree_str = self._line.replace(b'\\/', b'/').replace(b'\xc2\xa0', b'.').decode('utf-8')
#             tree_str = unidecode(tree_str)
#             tree = Tree.fromstring(tree_str)
#             dx = TreeKeeper.from_stan(tree)
#             self._words = words = tree.leaves()
#             token = np.asarray([w2i(w) for w in words])

#             layers_of_polars = []
#             layers_of_xtypes = []
#             for polars, xtypes in dx.trapezoid_gen(self._trapezoid_height, p2i, x2i):
#                 polars = np.asarray(polars)
#                 xtypes = np.asarray(xtypes)
#                 layers_of_polars.append(polars)
#                 layers_of_xtypes.append(xtypes)

#             factored = dict(token = token,
#                             polar = layers_of_polars,
#                             xtype = layers_of_xtypes)
#             self._factored = words, len(words), tree_str, factored
#         return self._factored
        
# # from data.multib import add_efficient_subs
# class PennWorker(Process):
#     def __init__(self, *args):
#         Process.__init__(self)
#         self._q_reader_fns_height_v2is_factors = args

#     def run(self):
#         (q, reader, fns, height, v2is, factors,
#          word_trace) = self._q_reader_fns_height_v2is_factors

#         for fn in fns:
#             for tree in reader.parsed_sents(fn):
#                 try:
#                     adjust_label(tree, word_trace = word_trace) # watch for ktb
#                 except:
#                     print(tree)
#                 # _, tree = add_efficient_subs(tree)
#                 words = tree.leaves()
#                 length = len(words)
#                 keeper = PennTreeKeeper(tree, v2is, height)
#                 factored = {f: keeper[f] for f in factors}
#                 if '(' in words or ')' in words:
#                     for i, word in enumerate(words):
#                         if word == '(':
#                             tree[tree.leaf_treeposition(i)] = '-LRB-'
#                         elif word == ')':
#                             tree[tree.leaf_treeposition(i)] = '-RRB-'
#                 results = words, length, str(tree), factored
#                 q.put(results)

# class StanWorker(Process):
#     def __init__(self, *args):
#         Process.__init__(self)
#         self._args = args

#     def run(self):
#         q, jobs, v2is, trapezoid_height = self._args
#         for line in jobs:
#             q.put(StanTreeKeeper(line, v2is, trapezoid_height).get())

