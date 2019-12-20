from data.backend import LengthOrderedDataset, np, torch
from data.delta import s_index, DeltaX, xtype_to_logits, preproc_cnf
from data.penn_types import select_and_split_corpus, SourcePool
from tqdm import tqdm
from itertools import zip_longest, count
from multiprocessing import Process, Queue
from time import sleep
# from data.delta import E_XDIM

fields = 'word', 'tag', 'ftag'
fieldx = 'label', 'xtype'
# FieldOrder = 'word', 'tag', 'label', 'xtype', 'ftag', 'length'

def split_v2is(v2is):
    _, w2i = v2is['word']
    _, t2i = v2is['tag']
    _, l2i = v2is['label']
    x2i = lambda x: xtype_to_logits(x, to_str = False)
    return w2i, t2i, l2i, x2i

class TreeKeeper:
    def __init__(self, tree, v2is, trapezoid_height):
        self._tree = tree
        self._v2is = v2is
        self._w_p = None
        self._factored = {}
        self._trapezoid_height = trapezoid_height

    def update_factored(self, factor, factored, words):
        self._w_p = factored['word'], factored['tag']
        self._factored[factor] = factored
        tree = self._tree
        for i, word in enumerate(words):
            if word == '(':
                tree[tree.leaf_treeposition(i)] = '('
            elif word == ')':
                tree[tree.leaf_treeposition(i)] = ')'

    def __getitem__(self, factor):
        if factor in self._factored:
            return self._factored[factor]

        w2i, t2i, l2i, x2i = self._v2is
        dx, _ = DeltaX.from_penn(self._tree, factor, do_preproc = False) # watch for keyaki arg wordtrace
        if self._w_p is None:
            word, tag = dx.word_tag(w2i, t2i)
            word = np.asarray(word)
            tag  = np.asarray(tag)
            self._w_p = word, tag
        else:
            word, tag = self._w_p
        
        layers_of_labels = []
        layers_of_xtypes = []
        for labels, xtypes in dx.trapezoid_gen(self._trapezoid_height, l2i, x2i):
            labels = np.asarray(labels)
            xtypes = np.asarray(xtypes)
            layers_of_labels.append(labels)
            layers_of_xtypes.append(xtypes)
        
        factored = dict(word = word,
                        tag  = tag,
                        label = layers_of_labels,
                        xtype = layers_of_xtypes)
        self._factored[factor] = factored
        return factored

    def __str__(self):
        s = f'Keeper with '
        return s + ', '.join(self._factored.keys()) + 'cached'

class WorkerX(Process):
    def __init__(self, *args):
        Process.__init__(self)
        self._q_reader_fns_height_v2is_factor = args

    def run(self):
        (q, reader, fns, height, v2is,
         factor) = self._q_reader_fns_height_v2is_factor
        
        for fn in fns:
            for tree in reader.parsed_sents(fn):
                preproc_cnf(tree)
                words = tree.leaves()
                length = len(words)
                keeper = TreeKeeper(tree, v2is, height)
                factored = keeper[factor]
                if '(' in words or ')' in words:
                    for i, word in enumerate(words):
                        if word == '(':
                            tree[tree.leaf_treeposition(i)] = '-LRB-'
                        elif word == ')':
                            tree[tree.leaf_treeposition(i)] = '-RRB-'
                results = words, length, str(tree), factored
                q.put(results)

from data.penn_types import Tree
class TrapezoidDataset(LengthOrderedDataset):
    def __init__(self,
                 trapezoid_height,
                 reader,
                 get_fnames,
                 data_split,
                 field_v2is,
                 paddings,
                 device,
                 factors,
                 min_len  = 0,
                 max_len  = None,
                 extra_text_helper = None,
                 num_threads = 0):

        v2is = split_v2is(field_v2is)
        text = []
        lengths = []
        keepers = []
        fnames = get_fnames(data_split)
        if num_threads < 1:
            from utils.types import num_threads
        # same with penn_type
        works = [[] for i in range(num_threads)]
        task_pool = SourcePool(works)
        q = Queue()
        for fname in fnames:
            work = task_pool()
            work.append(fname)
        del task_pool
        argmax_factor = max(factors, key = lambda x: factors[x])
        # import pdb; pdb.set_trace()
        for i in range(num_threads):
            w = WorkerX(q, reader, works[i], trapezoid_height, v2is, argmax_factor)
            w.start()
            works[i] = w

        with tqdm(desc = f'Receiving from {num_threads} threads ...') as qbar:
            while any(x.is_alive() for x in works):
                if q.empty():
                    sleep(0.01)
                else:
                    words, length, tree_str, factored = q.get()
                    text.append(words)
                    lengths.append(length)
                    keeper = TreeKeeper(Tree.fromstring(tree_str), v2is, trapezoid_height)
                    keeper.update_factored(argmax_factor, factored, words)
                    keepers.append(keeper)
                qbar.update(1)
            qbar.desc = f'{len(lengths)} TreesKeepers'

        heads = 'word tag label xtype'.split()
        extra_text_helper = None
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, device)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)

        self._paddings_device_height = paddings, device, trapezoid_height
        self._keepers = tuple(keepers)

    def at_idx(self, idx, factor):
        return self._keepers[idx][factor]

    def _collate_fn(self, batch):
        dtype = np.int32
        field_columns = {}
        paddings, device, height = self._paddings_device_height
        
        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, dtype)
                max_len = np.max(lengths)
                if paddings:
                    max_len += 2 # BOS and EOS
                    offsets = (max_len - lengths) // 2
                    field_columns['offset'] = offsets
                else:
                    field_columns['offset'] = np.zeros_like(lengths)
                full_triangular_len = s_index(max_len)
                tensor = lengths
            elif field in fields: # word or tags
                tensor = np.zeros([batch_size, max_len], dtype)
                for i_, (values, length) in enumerate(zip(column, lengths)):
                    if paddings:
                        start = offsets[i_]
                        end = start + length
                        bid, eid = paddings[field]
                        tensor[i_,    :start]  = bid
                        tensor[i_, start:end] = values
                        tensor[i_,      end:] = eid
                    else:
                        try:
                            tensor[i_, :length] = values
                        except:
                            import pdb; pdb.set_trace()
            else: # label or xtype
                tensor = np.zeros([batch_size, full_triangular_len], dtype = np.uint8)
                cumu_length = 0
                track_label = field == 'label'
                if track_label:
                    segments = []
                    mask_length = np.zeros([batch_size], dtype)
                    seg_length = np.zeros([batch_size, max_len], dtype)

                for l_, layer in enumerate(zip_longest(*column)):
                    max_layer_len = max(len(x) for x in layer if x is not None)
                    if paddings:
                        max_layer_len += 2
                    cumu_length += max_layer_len
                    l_start = full_triangular_len - cumu_length
                    l_end   = l_start + max_layer_len
                    if track_label:
                        segments.append(max_layer_len)
                    for i_, seq in enumerate(layer):
                        if seq is None:
                            continue
                        seq_len = len(seq)
                        if track_label:
                            mask_length[i_] += max_layer_len
                            seg_length[i_, -1 - l_] = seq_len
                        if paddings:
                            bid, eid = paddings[field]
                            start = l_start + offsets[i_]
                            end   = start + seq_len
                            tensor[i_, l_start:start] = bid
                            tensor[i_, start:end] = seq
                            tensor[i_, end:l_end] = eid
                        else:
                            end = l_start + seq_len
                            tensor[i_, l_start:end] = seq
                tensor = tensor[:, -cumu_length:]
            
            field_columns[field] = tensor

        for f, column in field_columns.items():
            field_columns[f] = torch.as_tensor(column, dtype = (None if f == 'xtype' else torch.long), device = device)

        segments.reverse()
        # height_segments = []
        # while segments:
        #     for i in count():
        #         if i % height == 0:
        #             height_segments.append(0)
        #         height_segments[-1] += segments.pop()
        #         if not segments:
        #             break
        # height_segments.reverse()
        field_columns['height'] = height
        field_columns['segment'] = segments
        field_columns['seg_length'] = seg_length[:, -len(segments):]
        field_columns['mask_length'] = mask_length

        return field_columns
