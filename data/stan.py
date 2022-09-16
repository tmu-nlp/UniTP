import torch
from os.path import join
from data.io import load_i2vs, get_fasttext
from data.vocab import VocabKeeper
from data.stan_types import get_sstb_trees
from data.dataset import LengthOrderedDataset, post_batch, pad_tag_like_nil, pad_label_like_nil
from data.continuous import Signal
from utils.param_ops import change_key
from data.mp import Process, Rush
from nltk.tree import Tree
Signal.set_binary()


class StanReader(VocabKeeper):
    def __init__(self, stan, extra_text_helper = None):
        self._corp = get_sstb_trees(stan.source_path)
        self._args = stan.condense_per, extra_text_helper

        vocab_path = stan.local_path
        i2vs = load_i2vs(vocab_path, 'word', 'polar')
        change_key(i2vs, 'word', 'token')

        def token_vec_fn():
            weights = get_fasttext(join(vocab_path, 'word.vec'))
            weights[0] = 0 # [nil ...]
            return weights
        super().__init__(('token', 'polar'), i2vs, {}, {}, weight_fn = token_vec_fn)#, **extra_fn)

    def batch(self, mode, batch_size, bucket_length,
              min_len        = 2,
              max_len        = None,
              sort_by_length = True):
        condense_per, extra_text_helper = self._args

        self.loaded_ds[mode] = ds = SentimentDataset(self.v2is, self._corp[mode], condense_per, min_len, max_len, extra_text_helper)        
        return post_batch(mode, ds, sort_by_length, bucket_length, batch_size)


class WorkerX(Process):
    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        i, q, lines, p2i, condense_per = self._args
        for line in lines:
            dx = Signal.from_sstb(Tree.fromstring(line))
            q.put((i, dx.serialize(False), dx.binary(l2i = p2i, every_n = condense_per)))
        q.put((i, len(lines)))


class SentimentDataset(LengthOrderedDataset):

    def __init__(self,
                 v2is, 
                 data,
                 condense_per,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None):

        w2i, p2i = v2is
        if condense_per < 1:
            heads = 'token', 'tree'
            label = None
        else:
            heads = 'token', 'polar', 'xtype'
            label = 'polar'

        length, token, signals, text, polar_x = [], [], [], [], []
        rush = Rush(WorkerX, data, p2i, condense_per)
        def receive(t, qbar):
            if isinstance(t[1], int):
                return t
            t, seri, px = t
            qbar.update(t)
            signal = Signal.instantiate(seri)
            length .append(signal.max_height)
            token  .append(signal.word_to_idx(w2i))
            text   .append(signal.word)
            signals.append(signal)
            polar_x.append(px)

        rush.mp_while(False, receive)

        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, w2i)

        super().__init__(heads, label, length, None, min_len, max_len, extra_text_helper)
        self._args = token, signals, polar_x, condense_per

    def at_idx(self, idx, factor, helper_outputs):
        token, signals, polar_x, condense_per = self._args
        sample = [token[idx]]
        if condense_per:
            sample.extend(polar_x[idx])
        else:
            sample.append(signals[idx].tree)
        return tuple(sample)

    def _collate_fn(self, batch, length, segment):
        max_token_len = length.max()
        field_columns = dict(length = length)
        indice_args = dict(device = self.device, dtype = torch.long)
        if segment is None:
            field_columns['tree'] = batch.tree
        else:
            batch_segment = segment.max(0)
            field_columns['condense_per'] = self._args[-1] # trigger traperzoid/triangle
            field_columns['polar'] = torch.as_tensor(pad_label_like_nil(batch.polar, batch_segment), **indice_args)
            field_columns['xtype'] = torch.as_tensor(pad_label_like_nil(batch.xtype, batch_segment), dtype = torch.uint8, device = self.device)
            field_columns['segment'] = segment
            field_columns['batch_segment'] = batch_segment
        field_columns['token'] = torch.as_tensor(pad_tag_like_nil(batch.token, max_token_len), **indice_args)
        return field_columns