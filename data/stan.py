import torch
from os.path import join
from data.io import load_i2vs, get_fasttext
from data.vocab import VocabKeeper
from data.stan_types import get_sstb_trees
from data.dataset import LengthOrderedDataset, post_batch, pad_tag_like_nil, pad_label_like_nil
from data.continuous import Signal
from utils.param_ops import change_key
from data import NIL
from data.mp import Process, mp_while
from nltk.tree import Tree
Signal.set_binary()


class StanReader(VocabKeeper):
    def __init__(self, stan, extra_text_helper = None):
        self._corp = get_sstb_trees(stan.source_path, stan.build_params._nested)
        self._args = stan, extra_text_helper

        vocab_path = stan.local_path
        i2vs = load_i2vs(vocab_path, 'word', 'polar')
        change_key(i2vs, 'word', 'token')
        oovs = {}
        if stan.neutral_nil:
            polar = i2vs['polar']
            assert polar.pop(0) == NIL
            oovs['polar'] = polar.index('2')

        def token_vec_fn():
            weights = get_fasttext(join(vocab_path, 'word.vec'))
            weights[0] = 0 # [nil ...]
            return weights
        super().__init__(('token', 'polar'), i2vs, oovs, {}, weight_fn = token_vec_fn)#, **extra_fn)

    def binary(self, mode, condense_per, batch_size, bucket_length,
               min_len        = 2,
               max_len        = None,
               sort_by_length = True):
        _, extra_text_helper = self._args

        self.loaded_ds[mode] = ds = SentimentDataset(self.v2is, self._corp[mode], 
            condense_per, min_len, max_len, extra_text_helper) # even test set need orientation.
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)


class WorkerX(Process):
    estimate_total = False

    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        i, q, lines, p2i, condense_per = self._args
        for line in lines:
            dx = Signal.from_sstb(Tree.fromstring(line))
            px = None if condense_per is None else dx.binary(l2i = p2i, every_n = condense_per)
            q.put((i, dx.serialize(False), px))
        q.put((i, len(lines), 0))


class SentimentDataset(LengthOrderedDataset):

    def __init__(self,
                 v2is, 
                 data,
                 condense_per,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        w2i, p2i = v2is
        if condense_per is None:
            heads = 'tree', 'token'
            label = None
        else:
            heads = 'tree', 'token', 'polar', 'xtype'
            label = 'polar'

        length, token, signals, text, polar_x = [], [], [], [], []
        def receive(t, qbar):
            if isinstance(t[1], int):
                return t
            tid, seri, px = t
            qbar.update(tid)
            signal = Signal.instantiate(seri)
            length .append(signal.max_height)
            token  .append(signal.word_to_idx(w2i))
            text   .append(signal.word)
            signals.append(signal)
            polar_x.append(px)
        mp_while(WorkerX, data, receive, p2i, condense_per)

        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, w2i)

        super().__init__(heads, label, length, None, min_len, max_len, extra_text_helper)
        self._args = token, signals, polar_x, condense_per

    def at_idx(self, idx, factor, helper_outputs):
        token, signals, polar_x, condense_per = self._args
        sample = [signals[idx].tree, token[idx]]
        if condense_per:
            sample.extend(polar_x[idx])
        return tuple(sample)

    def _collate_fn(self, batch, length, segment):
        max_token_len = length.max()
        field_columns = dict(length = length, tree = batch.tree)
        indice_args = dict(device = self.device, dtype = torch.long)
        if segment is not None:
            batch_segment = segment.max(0)
            field_columns['condense_per'] = self._args[-1] # trigger traperzoid/triangle
            field_columns['polar'] = torch.as_tensor(pad_label_like_nil(batch.polar, batch_segment), **indice_args)
            field_columns['xtype'] = torch.as_tensor(pad_label_like_nil(batch.xtype, batch_segment), dtype = torch.uint8, device = self.device)
            field_columns['segment'] = segment
            field_columns['batch_segment'] = batch_segment
        field_columns['token'] = torch.as_tensor(pad_tag_like_nil(batch.token, max_token_len), **indice_args)
        return field_columns