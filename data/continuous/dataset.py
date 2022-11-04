import torch
from data.continuous import Signal
from data.continuous.binary import X_RGT
from data.dataset import binary_signals, checkin_cache
from data.dataset import LengthOrderedDataset, np, read_signals
from data.dataset import pad_tag_like_nil, pad_tag_like_bos_eos
from data.dataset import pad_label_like_nil, pad_label_like_bos_eos
from data.dataset import erect_joint_more, erect_split_more, fill_bool_tensor

class ContinuousDataset(LengthOrderedDataset):
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

        w2i, t2i, l2i = v2is
        (length, token, tag, signals,
         text) = read_signals(w2i, t2i, fileids, reader, Signal, from_fn, esub, char_as_token)
        heads = 'tree', 'token'
        if factor is None:
            label = extra = signal_kwargs = None
        else:
            if binary:
                _heads = 'tag', 'label', 'xtype'
                factor, extra = self.reset_binary_factor(factor, esub, msub, initialize = len(length))
                signal_kwargs = dict(l2i = l2i, every_n = b_condense_or_m_joint)
            else:
                _heads = 'tag', 'label', 'chunk'
                factor, extra = self.reset_multib_factor(esub, msub, initialize = len(length))
                signal_kwargs = dict(l2i = l2i, joint = b_condense_or_m_joint)
            heads = heads + _heads
            label = 'label'
        if char_as_token:
            heads = heads + ('char_segment',)

        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, w2i)

        super().__init__(heads, label, length, factor, min_len, max_len, extra_text_helper)
        self._args = token, tag, signals, signal_kwargs, char_as_token, paddings, binary, b_condense_or_m_joint, self_check_i2vs, extra

    def reset_multib_factor(self, esub, msub, *, initialize = False):
        esub = {0: 1 - esub, 1: esub}
        if initialize:
            return esub, tuple([None, None] for _ in range(initialize)) if msub == 0 else msub
        if msub > 0:
            self._args = self._args[:-1] + (msub,)
        elif isinstance(self._args[-1], int):
            self._args = self._args[:-1] + (tuple([None, None] for _ in range(len(self._args[0]))),)
        super()._reset_factors(esub)

    def at_idx(self, idx, factor, helper_outputs):
        token, tag, signals, signal_kwargs, char_as_token, _, binary, b_condense_or_m_joint, _, extra = self._args
        signal = signals[idx]
        sample = [signal.tree, token[idx]]
        if extra is not None:
            sample.append(tag[idx])
            if binary:
                lb, sp = binary_signals(factor, idx, extra, lambda frac, esub, msub = 0: signal.binary(frac, esub, msub, **signal_kwargs))
            elif isinstance(extra, tuple):
                cache = extra[idx]
                if cache[factor] is None:
                    checkin_cache(cache, factor, signal.multib(factor, **signal_kwargs))
                lb, sp = cache[factor]
            else:
                lb, sp = signal.multib(factor, extra, **signal_kwargs) # more_sub
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
                char_joint = erect_joint_more(batch_char_segment, batch.char_segment, max_token_len, 0)
            else:
                char_split = erect_split_more(batch.char_segment, [max_token_len], 0)
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
                    fill_bool_tensor(char_joint, sig, False, indice_args)
                else: # split (load slower)
                    sig = torch.zeros(len(length), max_token_len + 1, dtype = torch.bool, device = self.device)
                    fill_bool_tensor(char_split, sig, True, indice_args)
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
                    fill_bool_tensor(char_joint + erect_joint_more(segment, batch.chunk, batch_segment, sig_offset), sig, False, indice_args)
                else: # split (load slower)
                    sig = torch.zeros(len(length), sig_size, dtype = torch.bool, device = self.device)
                    fill_bool_tensor(char_split + erect_split_more(batch.chunk, batch_segment, sig_offset), sig, True, indice_args)
                field_columns['chunk'] = sig[:, :-2] # top 2 are stable useless ones
                
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