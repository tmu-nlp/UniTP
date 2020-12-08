from data.backend import LengthOrderedDataset, np, torch
from utils.shell_io import byte_style
from data.m_ary import MAryX, Tree, draw_str_lines
from tqdm import tqdm
from itertools import zip_longest
from data.io import distribute_jobs
from multiprocessing import Process, Queue
from utils.file_io import DelayedKeyboardInterrupt
from time import sleep
from sys import stderr
from collections import defaultdict, Counter

class MAryWorker(Process):
    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        q, trees, field_v2is = self._args
        for tree in trees:
            mtree = MAryX(tree)
            signals = mtree.signals(*field_v2is)
            q.put((mtree.words, signals, tuple(None in s for s in signals[:2])))
            # if field_v2is['label']._flag:
            #     print('\n'.join(draw_str_lines(Tree.fromstring(_tree))))
            #     print('\n'.join(draw_str_lines(tree)))
            #     field_v2is['label']._flag = False
            #     import pdb; pdb.set_trace()


class MAryDataset(LengthOrderedDataset):
    def __init__(self,
                 mode,
                 samples,
                 field_v2is,
                 device,
                 min_len = 0,
                 max_len = None,
                 extra_text_helper = None,
                 num_threads = 0):

        samples = [x for m, x in samples if m == mode]
        if num_threads < 1:
            from utils.types import num_threads
        works = distribute_jobs(samples, num_threads)
        q = Queue()
        for i in range(num_threads):
            w = MAryWorker(q, works[i], tuple(field_v2is[x][1] for x in ('token', 'tag', 'label')))
            w.start()
            works[i] = w

        text = []
        lengths = []
        signals = []
        oov_errors = []
        with tqdm(desc = 'Load ' + byte_style(mode.title().ljust(5, '-') + 'Set', '2') + f' from {num_threads} threads', total = len(samples)) as qbar:
            try:
                while any(x.is_alive() for x in works):
                    if q.empty():
                        sleep(0.00001)
                    else:
                        words, signal, oov = q.get()
                        qbar.update(1)
                        if any(oov):
                            oov_errors.append((oov, words))
                        else:
                            text.append(words)
                            lengths.append(len(words))
                            signals.append(signal)
            except KeyboardInterrupt as ex:
                with DelayedKeyboardInterrupt(ignore = True):
                    for x in works:
                        x.kill()
                raise ex

            if oov_errors:
                qbar.desc += ', ended with ' + byte_style(f'{len(oov_errors)} OOV errors', '3')

        if oov_errors:
            oov_length = defaultdict(int)
            oov_words = Counter()
            oov_tags = 0
            for (word_oov, tag_oov), words in oov_errors:
                if word_oov:
                    oov_length[len(words)] += 1
                    oov_words += Counter(words)
                if tag_oov:
                    oov_tags += 1

            error_string = ''
            if oov_words:
                error_string += ' Len:'
                error_string += ' '.join(f'{l}/{c}' for l,c in oov_length.items())
                error_string += ' | OOV word(s): '
                error_string += ' '.join(w+f'/{c}' for w,c in oov_words.items())
            if oov_tags: error_string += f'; {oov_tags} OOV tag(s)'
            print(error_string, file = stderr)

        heads = 'token', 'tag', 'label', 'fence'
        self._signals_heads = signals, heads
        if extra_text_helper:
            extra_text_helper = extra_text_helper(text, device)
        super().__init__(heads, lengths, None, min_len, max_len, extra_text_helper)
        self._device = device

    def at_idx(self, idx, factor, length):
        assert factor is None
        signals, heads = self._signals_heads
        sample = {h:s for h, s  in zip(heads, signals[idx])}
        sample['length'] = length
        return sample

    def _collate_fn(self, batch):
        field_columns = {}
        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                tensor = lengths = np.asarray(column, np.int32)
                max_len = np.max(lengths)
            elif field in ('token', 'tag'):
                tensor = np.zeros([batch_size, max_len], np.int32)
                for i, (values, length) in enumerate(zip(column, lengths)):
                    tensor[i, :length] = values
            elif field == 'label':
                segment = []
                seg_len = []
                for layer in zip_longest(*column, fillvalue = []):
                    sl = [len(seq) for seq in layer]
                    segment.append(max(sl))
                    seg_len.append(sl)
                field_columns['segment'] = torch.tensor(segment) # CPU
                field_columns['seg_length'] = torch.tensor(seg_len).transpose(0, 1) # CPU
                tensor = fill_layers(column, segment, np.int32, torch.long)
            else:
                tensor_seg = [x + 1 for x in segment[1:]]
                tensor = fill_layers(column, tensor_seg, np.int32, torch.long, 1)
            field_columns[field] = torch.as_tensor(tensor, dtype = torch.long, device = self._device)
        return field_columns

def fill_layers(sample_layers, tensor_seg, np_dtype, torch_dtype, last_elem = None):
    batch_size = len(sample_layers)
    tensor = np.zeros([batch_size, sum(tensor_seg)], dtype = np_dtype)
    start = 0
    last_elem_offset = 0 if last_elem is None else 1 # label vs. fence
    for seg_len, layer in zip(tensor_seg, zip_longest(*sample_layers, fillvalue = [])):
        end = start + seg_len
        for bid, seq in enumerate(layer):
            if seq:
                seq_len = len(seq)
                tensor[bid, start:start + seq_len] = seq
                if last_elem is None and seq_len == 1:
                    last_elem = seq[0]
            else:
                tensor[bid, start + last_elem_offset] = last_elem
        start = end
    return tensor