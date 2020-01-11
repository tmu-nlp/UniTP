#!/usr/bin/env python

import numpy as np
from collections import namedtuple, Counter, defaultdict
from itertools import count
from os.path import join, isfile, getsize, expanduser, basename
from os import listdir, remove
from data.delta import s_index, t_index, NIL
from data.delta import get_tree_from_triangle, explain_warnings, explain_one_error
import sys, pdb
from nltk.tree import Tree
from utils.math_ops import isqrt
from utils.pickle_io import pickle_load, pickle_dump

IOHead = namedtuple('IOHead', 'offset, length, word, tag, label, right, tree, segment, seg_length')
IOData = namedtuple('IOData', 'offset, length, word, tag, label, right, tree, segment, seg_length, mpc_word, mpc_phrase, warning, scores, tag_score, label_score, split_score, summary')

inf_none_gen = (None for _ in count())

def triangle_to_layers(data, *size_offset_length_vocab):
    if size_offset_length_vocab:
        size, offset, length, vocab = size_offset_length_vocab
    else:
        length, offset = t_index(len(data))
        assert offset == 0
        size = length
        vocab = None

    pad_len = size - length
    layers = []
    for level in range(size):
        seq_len = level - pad_len
        if seq_len < 0:
            continue
        start = s_index(level) + offset
        end   = start + seq_len + 1
        layer = data[start:end]
        if vocab:
            layer = tuple(vocab(x) for x in layer)
        layers.append(layer)
    return layers

def trapezoid_to_layers(data, segments, seg_length, vocab = None, offset = 0):
    layers  = []
    l_end   = len(data)
    seg_len = list(zip(segments, seg_length))
    while seg_len:
        size, seq_len = seg_len.pop()
        l_start = l_end - size + offset
        layer = data[l_start:l_start + seq_len]
        if vocab:
            layer = tuple(vocab(x) for x in layer)
        layers.append(layer)
        if seq_len == 1: break
        l_end -= size
    return layers

def __before_to_tree_(offset, length, words, tags, labels, rights, segments, seg_lengths, vocabs):
    word_layer, tag_layer, label_vocab = __before_to_seq(offset, length, words, tags, labels, vocabs)
    label_layers = trapezoid_to_layers(labels, segments, seg_lengths, label_vocab)
    right_layers = trapezoid_to_layers(rights, segments, seg_lengths,        None)
    return word_layer, tag_layer, label_layers, right_layers

def __before_to_seq(offset, length, words, tags, labels, vocabs):
    word_layer      = tuple(vocabs.word[w] for w in words[offset:offset+length])
    if tags is not None: # label_mode
        tag_layer   = tuple(vocabs.tag[t]  for t in tags [offset:offset+length])
        label_vocab = vocabs.label.__getitem__
    else:
        tag_layer = None
        label_vocab = lambda x: NIL if x < 0 else vocabs.label[x]
    return word_layer, tag_layer, label_vocab

def __before_to_tree(offset, length, words, tags, labels, rights, vocabs):
    size = len(words)
    word_layer, tag_layer, label_vocab = __before_to_seq(offset, length, words, tags, labels, vocabs)
    label_layers = triangle_to_layers(labels, size, offset, length, label_vocab)
    right_layers = triangle_to_layers(rights, size, offset, length,        None)
    label_layers.reverse()
    right_layers.reverse()
    return word_layer, tag_layer, label_layers, right_layers

def head_to_tree_(offset, length, words, tags, labels, rights, seg_lengths, segments, vocabs):
    args = __before_to_tree_(offset, length, words, tags, labels, rights, segments, seg_lengths, vocabs)
    tree, warn = get_tree_from_triangle(*args)
    assert len(warn) == 0
    return tree

def head_to_tree(offset, length, words, tags, labels, rights, vocabs):
    tree, warn = get_tree_from_triangle(*__before_to_tree(offset, length, words, tags, labels, rights, vocabs))
    assert len(warn) == 0
    return tree

def data_to_tree_(offset, length, words, tags, labels, rights, seg_lengths, segments, vocabs,
                  return_warnings = False,
                  on_warning      = None,
                  on_error        = None,
                  error_prefix    = ''):
    return __after_to_tree(*__before_to_tree_(offset, length, words, tags, labels, rights, segments, seg_lengths, vocabs),
                           return_warnings,
                           on_warning,
                           on_error,
                           error_prefix)

# demands:
# 1. want to know whether there are warnings or errors and a safe results (e.g. individual visualization, calc whole scores)
# 2. suppress all the warnings and error (output to stderr), just get a safe result
# [4: 261], [5: 197], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683； relay
# [4: 598], [5: 998], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683： keep
def data_to_tree(offset, length, words, tags, labels, rights, vocabs,
                 return_warnings = False,
                 on_warning      = None,
                 on_error        = None,
                 error_prefix    = ''):
    return __after_to_tree(*__before_to_tree(offset, length, words, tags, labels, rights, vocabs),
                           return_warnings,
                           on_warning,
                           on_error,
                           error_prefix)

def __after_to_tree(word_layer, tag_layer, label_layers, right_layers,
                    return_warnings = False,
                    on_warning      = None,
                    on_error        = None,
                    error_prefix    = ''):
    try:
        tree, warnings = get_tree_from_triangle(word_layer, tag_layer, label_layers, right_layers)
    except ValueError as e:
        error, last_layer, warnings = e.args
        if callable(on_error):
            on_error(error_prefix, explain_one_error(error))
        tree = Tree('S', [x for x in last_layer if x]) # Trust the model: TODO report failure rate
        warnings.append(error)
    if warnings and callable(on_warning) and tag_layer is not None:
        on_warning(explain_warnings(warnings, label_layers, tag_layer))
    if return_warnings: # [:, 2] > 8 is error
        warnings = np.asarray(warnings, dtype = np.int8)
        warnings.shape = (-1, 3)
        return tree, warnings
    return tree

# def convert_batch(h, d, num_word, vocabs, fh, fd):

#     for i, l in enumerate(h.len):
#         if fh is not None:
#             tree = head_to_tree(h.word[i], h.tag[i], h.label[i], l, h.left[i], vocabs)
#             print(' '.join(str(tree).split()), file = fh)
#         tree, warnings = data_to_tree(h.word[i], d.tag[i], _label(i), l, _left(i), vocabs, return_warnings = True)
#         if fd is not None:
#             print(' '.join(str(tree).split()), file = fd)
#         yield i, l, warnings

def set_vocab(fpath, vocabs, model_vocab_size = None, fname = 'vocabs.pkl'):
    assert isinstance(vocabs, dict)
    fname = join(fpath, fname)
    if isfile(fname):
        return False
    pickle_dump(fname, vocabs)
    return True

from contextlib import ExitStack
def set_head(fpath, batch_id, size,
             offset, length, word, tag, label, right,
             trapezoid_info,
             vocabs, fhtree, vfname = 'vocabs.pkl', bin_width = 10):
    old_tag = tag

    if tag is None:
        tag = inf_none_gen

    if trapezoid_info is None:
        segment = seg_length = None
        func_args = zip(offset, length, word, tag, label, right)
    else:
        segment, seg_length = trapezoid_info
        func_args = zip(offset, length, word, tag, label, right, seg_length)

    trees = []
    for args in func_args:
        if trapezoid_info is None:
            tree = str(head_to_tree(*args, vocabs))
        else:
            tree = str(head_to_tree_(*args, segment, vocabs))
        tree = ' '.join(tree.split())
        print(tree, file = fhtree)
        trees.append(tree)

    if fpath:

        ftrees = {}
        fhead = join(fpath, f'head.{batch_id}.tree')

        with ExitStack() as stack, open(fhead, 'w') as fh:
            for wlen, tree in zip(length, trees):
                wbin = wlen // bin_width
                if wbin in ftrees:
                    fw = ftrees[wbin]
                else:
                    fw = open(join(fpath, f'head.bin_{wbin}.tree'), 'a')
                    ftrees[wbin] = stack.enter_context(fw)
                print(tree, file = fw)
                print(tree, file = fh)

        assert isfile(join(fpath, vfname))
        fname = join(fpath, f'head.{batch_id}_{size}.pkl')
        head  = IOHead(offset, length, word, old_tag, label, right, trees, segment, seg_length)
        pickle_dump(fname, head)
        return ftrees.keys()

from utils.shell_io import parseval, rpt_summary
def set_data(fpath, batch_id, size, epoch,
             offset, length, word, tag, label, right, mpc_word, mpc_phrase,
             tag_score, label_score, split_score,
             trapezoid_info,
             vocabs, fdtree, on_error = None, evalb = None, bin_width = 10):

    tree_kwargs = dict(return_warnings = True, on_error = on_error)
    error_prefix = f'  [{batch_id} {epoch}'

    old_tag   = tag
    old_tag_s = tag_score
    if tag is None: tag = inf_none_gen
    trees = []
    batch_warnings = []
    if trapezoid_info is None:
        segment = seg_length = None
        func_args = zip(offset, length, word, tag, label, right)
    else:
        segment, seg_length = trapezoid_info
        func_args = zip(offset, length, word, tag, label, right, seg_length)
    for i, args in enumerate(func_args):
        tree_kwargs['error_prefix'] = error_prefix + f']-{i} len={args[1]}'
        if trapezoid_info is None:
            tree, warnings = data_to_tree(*args, vocabs, **tree_kwargs)
        else:
            tree, warnings = data_to_tree_(*args, segment, vocabs, **tree_kwargs)
        tree = str(tree)
        tree = ' '.join(tree.split())
        trees.append(tree)
        print(tree, file = fdtree) # TODO use stack to protect opened file close and delete
        batch_warnings.append(warnings)

    if fpath:
        fdata = join(fpath, f'data.{batch_id}.tree')
        with open(fdata, 'w') as fw:
            for tree in trees:
                print(tree, file = fw)
                
        fhead = join(fpath, f'head.{batch_id}.tree')
        proc = parseval(evalb, fhead, fdata)
        idv, smy = rpt_summary(proc.stdout.decode(), True, True)

        fhead = f'head.{batch_id}_{size}.pkl'
        assert isfile(join(fpath, fhead)), f"Need a head '{fhead}'"
        fdata = join(fpath, f'data.{batch_id}_{epoch}.pkl')
        data = IOData(offset, length, word, tag, label, right, trees,
                      segment, seg_length, mpc_word, mpc_phrase,
                      batch_warnings, idv, tag_score, label_score, split_score, smy)
        pickle_dump(fdata, data)

        fname = join(fpath, 'summary.pkl')
        if isfile(fname):
            summary = pickle_load(fname)
        else:
            summary = {}
        summary[(batch_id, epoch)] = smy
        pickle_dump(fname, summary)

        if isinstance(bin_width, int):
            ftrees = {}
            with ExitStack() as stack:
                for wlen, tree in zip(length, trees):
                    wbin = wlen // bin_width
                    if wbin in ftrees:
                        fw = ftrees[wbin]
                    else:
                        fw = open(join(fpath, f'data.bin_{wbin}.tree'), 'a')
                        ftrees[wbin] = stack.enter_context(fw)
                    print(tree, file = fw)

        return smy['F1']

def calc_stan_accuracy(folder, bid, e_major, e_minor, on_error):
    
    numerators   = [0,0,0,0]
    denominators = [0,0,0,0]
    neg_set = '01'
    pos_set = '34'
    tree_error_prefix = f'  [{bid}, {e_major}, {e_minor}]'

    sents = []
    if bid is None:
        hfname = 'tree.head'
        dfname = f'tree.data.{e_major}_{e_minor}'
    else:
        hfname = f'head.{bid}.tree'
        dfname = f'data.{bid}_{e_major}_{e_minor}.tree'
    with open(join(folder, hfname)) as fh,\
         open(join(folder, dfname)) as fd:
        for i, head, data in zip(count(), fh, fd):

            warnings = []
            _numerators   = [0,0,0,0]
            _denominators = [0,0,0,0]
            head = Tree.fromstring(head)
            data = Tree.fromstring(data)
            seq_len = len(head.leaves())
            if seq_len != len(data.leaves()):
                warnings.append(f'lengths do not match vs. {len(data.leaves())}')
            for ith in range(seq_len):
                if head.leaf_treeposition(ith) != data.leaf_treeposition(ith):
                    warnings.append(f'shapes do not match at {ith}-th leaf')
                    break
            if warnings:
                error_prefix = tree_error_prefix + f'.{i} len={seq_len}'
                on_error(error_prefix, warnings[-1])
            gr = head.label()
            pr = data.label()
            _denominators[0] += 1
            if gr == pr:
                _numerators[0] += 1
            if gr != '2':
                _denominators[1] += 1
                if (gr in pos_set and pr in pos_set) or (gr in neg_set and pr in neg_set):
                    _numerators[1] += 1
            
            for gt, pt in zip(head.subtrees(), data.subtrees()):
                _denominators[2] += 1
                gt = gt.label()
                pt = pt.label()
                if gt == pt:
                    _numerators[2] += 1
                if gt != '2':
                    _denominators[3] += 1
                    if (gt in pos_set and pt in pos_set) or (gt in neg_set and pt in neg_set):
                        _numerators[3] += 1
            scores = []
            for n,d in  zip(_numerators, _denominators):
                scores.append(n/d*100 if d else float('nan'))
            sents.append(scores)

            for i in range(4):
                numerators[i]   += _numerators[i]
                denominators[i] += _denominators[i]

    scores = []
    for n,d in  zip(numerators, denominators):
        scores.append(n/d*100 if d else float('nan'))
    # 0: fine, 1: np, 2: fine_root, 3: np_root
    return sents, scores, (numerators, denominators)
# dpi_value     = master.winfo_fpixels('1i')
# master.tk.call('tk', 'scaling', '-displayof', '.', dpi_value / 72.272)
# screen_shape = master.winfo_screenwidth(), master.winfo_screenheight()
# master.geometry("%dx%d+%d+%d" % (canvas_shape + tuple(s/2-c/2 for s,c in zip(screen_shape, canvas_shape))))

try:
    from utils.gui import *
    desktop = True
except ImportError:
    desktop = False

if desktop:
    def _font(x):
        font_name, font_size = x.split()
        font_size = int(font_size)
        assert font_size > 0, 'font size should be positive'
        return font_name, font_size

    def _ratio(x):
        x = float(x) if '.' in x else int(x)
        assert x > 0, 'should be positive'
        return x

    def _frac(x):
        x = float(x)
        assert 0 < x < 1, 'should be a fraction'
        return x

    def _size(x):
        x = int(x)
        assert x > 0, 'should be positive'
        return x

    def _dim(x):
        x = int(x)
        assert 0 < x < 10, 'should be in [1, 9]'
        return x

    def _curve(x):
        x = x.strip()
        assert 'x' in x, 'should contain variable x'
        if ':' not in x:
            x = 'lambda x:' + x
        curve = eval(x)
        assert callable(curve), 'shoule be a function'
        assert isinstance(curve(0), float), 'should return a float number'
        return curve

    BoolList = namedtuple('BoolList', 'delta_shape, show_errors, show_paddings, show_nil, dark_background, inverse_brightness, align_coord, show_color, statistics')
    CombList = namedtuple('CombList', 'curve, dash, gauss, picker, spotlight')
    DynamicSettings = namedtuple('DynamicSettings', BoolList._fields + tuple('apply_' + f for f in CombList._fields) + CombList._fields)
    NumbList = namedtuple('NumbList', 'font, pc_x, pc_y, line_width, word_width, word_height, xy_ratio, histo_width, scatter_width')
    numb_types = NumbList(_font, _dim, _dim, _ratio, _size, _size, _ratio, _size, _size)
    comb_types = CombList(_curve, _frac, _ratio, _ratio, _ratio)
    PanelList = namedtuple('PanelList', 'hide_listboxes, detach_viewer')
    navi_directions = '⇤↑o↓⇥'

    from colorsys import hsv_to_rgb, hls_to_rgb
    from tempfile import TemporaryDirectory
    from nltk.draw import TreeWidget
    from nltk.draw.util import CanvasFrame
    from time import time, sleep
    from math import exp, sqrt, pi
    from functools import partial
    from data.delta import warning_level
    from utils.param_ops import more_kwargs, HParams
    from utils.file_io import path_folder
    from concurrent.futures import ProcessPoolExecutor
    # from multiprocessing import Pool

    class PathWrapper:
        def __init__(self, fpath, sftp):
            fpath, folder = path_folder(fpath)
            self._folder = '/'.join(folder[-2:])
            if sftp is not None:
                sftp.chdir(fpath)
                fpath = TemporaryDirectory()
            self._fpath_sftp = fpath, sftp

        def join(self, fname):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                f = join(fpath.name, fname)
                if fname not in listdir(fpath.name):
                    print('networking for', fname)
                    start = time()
                    sftp.get(fname, f)
                    print('transfered %d KB in %.2f sec.' % (getsize(f) >> 10, time() - start))
                else:
                    print('use cached', fname)
                return f
            return join(fpath, fname)

        @property
        def base(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                return fpath.name
            return fpath

        @property
        def folder(self):
            return self._folder

        def listdir(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                return sftp.listdir()
            return listdir(fpath)

        def __del__(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                # fpath.cleanup() # auto cleanup
                sftp.close()

    get_batch_id = lambda x: int(x[5:-4].split('_')[0])
    R = namedtuple('R', 'head, sent, pool')

    class TreeExplorer(Frame):
        def __init__(self,
                     root,
                     fpath,
                     initial_bools = BoolList(True, False, False, False, False, True, False, True, False),
                     initial_numbs = NumbList('System 15', (1, 1, 9, 1), (2, 1, 9, 1), (4, 2, 10, 2), (80, 60, 200, 5), (22, 12, 99, 2), (0.9, 0.5, 2, 0.1), (60, 50, 120, 10), (60, 50, 120, 10)),
                     initial_panel = PanelList(False, False),
                     initial_combs = CombList((True, 'x ** 0.5'), (False, (0.5, 0.1, 0.9, 0.1)), (True, (0.04, 0.01, 0.2, 0.01)), (True, (0.2, 0.1, 0.9, 0.1)), (False, (200, 100, 500, 100)))):
            vocabs = fpath.join('vocabs.pkl')
            if isfile(vocabs):
                self._vocabs = HParams(pickle_load(vocabs))
            else:
                raise ValueError(f"The folder should at least contains a vocab file '{vocabs}'")

            self._fpath_heads = fpath, None
            self._last_init_time = None
            self._rpt = R(defaultdict(list), {}, None)

            super().__init__(root)
            self.master.title(fpath.folder)
            headbox = Listbox(self, relief = SUNKEN)
            sentbox = Listbox(self, relief = SUNKEN)
            self._boxes = headbox, sentbox
            self.initialize_headbox()
            self._sent_cache = {}
            self._cross_warnings = {}

            headbox.bind('<<ListboxSelect>>', self.read_listbox)
            sentbox.bind('<<ListboxSelect>>', self.read_listbox)

            pack_kwargs = dict(padx = 10, pady = 5)
            intr_kwargs = dict(pady = 2)

            control = [1 for i in range(len(initial_numbs))]
            control[0] = dict(char_width = 17)
            control_panel = Frame(self, relief = SUNKEN)
            ckb_panel = Frame(control_panel)
            etr_panel = Frame(control_panel)
            self._checkboxes = make_namedtuple_gui(make_checkbox, ckb_panel, initial_bools, self._change_bool,   **intr_kwargs)
            self._entries    = make_namedtuple_gui(make_entry,    etr_panel, initial_numbs, self._change_string, control)
            ckb_panel.pack(side = TOP, fill = X, **pack_kwargs)
            etr_panel.pack(side = TOP, fill = X, **pack_kwargs) # expand means to pad
            self._last_bools = initial_bools
            self._last_numbs = NumbList(*(x[0] if isinstance(x, tuple) else x for x in initial_numbs))

            comb_panel = Frame(control_panel)
            self._checkboxes_entries = make_namedtuple_gui(make_checkbox_entry, comb_panel, initial_combs, (self._change_bool, self._change_string), (2, 1, 1, 1, 1))
            comb_panel.pack(side = TOP, fill = X, **pack_kwargs)
            self._last_combs = CombList(*((x[0], x[1][0]) if isinstance(x[1], tuple) else x for x in initial_combs))
            
            view_panel = Frame(control_panel)
            self._panels = make_namedtuple_gui(make_checkbox, view_panel, initial_panel, self.__update_viewer, **intr_kwargs)
            view_panel.pack(side = TOP, fill = X, **pack_kwargs)
            self._last_panel_bools = initial_panel

            navi_panel = Frame(control_panel)
            navi_panel.pack(fill = X)
            btns = tuple(Button(navi_panel, text = p) for p in navi_directions)
            for btn in btns:
                btn.bind("<Button-1>", self._navi)
                btn.pack(side = LEFT, fill = X, expand = YES)
            self._navi_btns = btns

            btn_panel = Frame(control_panel)
            btn_panel.pack(side = TOP, fill = X) # no need to expand, because of st? can be bottom
            st = Button(btn_panel, text = 'Show Trees', command = self._show_one_tree )
            sa = Button(btn_panel, text = '♾', command = self._show_all_trees)
            sc = Button(btn_panel, text = '◉', command = self._save_canvas   )
            st.pack(side = LEFT, fill = X, expand = YES)
            sa.pack(side = LEFT, fill = X)
            sc.pack(side = LEFT, fill = X)
            self._panel_btns = st, sa, sc

            self._control_panel = control_panel
            control_panel.bind('<Key>', self.shortcuts)
            headbox.bind('<Key>', self.shortcuts)
            sentbox.bind('<Key>', self.shortcuts)

            self._viewer = None
            self._selected = tuple()
            self._init_time = 0
            self.__update_layout(True, True)

        def initialize_headbox(self):
            fpath = self._fpath_heads[0]
            headbox = self._boxes[0]
            headbox.delete(0, END)
            fnames = fpath.listdir()
            heads = [f for f in fnames if f.startswith('head.') and f.endswith('.pkl')]
            heads.sort(key = get_batch_id)
            if len(heads) == 0:
                raise ValueError("'%s' is an invalid dir" % fpath.base)

            if 'summary.pkl' in fnames:
                summary = pickle_load(fpath.join('summary.pkl'))
                from math import isnan
            else:
                summary = {}
            for h in heads:
                fscores = []
                bid = get_batch_id(h)
                
                for (batch_id, epoch), smy in summary.items():
                    if batch_id == bid:
                        if not isnan(smy['F1']):
                            fscores.append(smy['F1'])

                if len(fscores) == 0:
                    fscores = ''
                elif len(fscores) == 1:
                    fscores = f'  {fscores[0]}'
                else:
                    fscores = f'  ≤{max(fscores)}'
                headbox.insert(END, h[5:-4].replace('_', '.\t<') + fscores)
            self._fpath_heads = fpath, heads
            self._last_init_time = time()

        # def __del__(self):
        #     if self._rpt.alabelc:
        #         print(f'terminate receiving {len(self._rpt.alabelc)} rpt files')
        #         self._rpt.pool.terminate()

        def __update_layout(self, hide_listboxes_changed, detach_viewer_changed):
            headbox, sentbox = self._boxes
            control_panel = self._control_panel
            viewer = self._viewer

            if detach_viewer_changed:
                if viewer and viewer.winfo_exists():
                    self._init_time = viewer.time
                    viewer.destroy()
                # widget shall be consumed within a function, or they will be visible!
                master = Toplevel(self) if self._last_panel_bools.detach_viewer else self
                viewer = ThreadViewer(master, self._vocabs, self._change_title)
                viewer.bind('<Key>', self.shortcuts)
                self._viewer = viewer

            if hide_listboxes_changed:
                if self._last_panel_bools.hide_listboxes:
                    headbox.pack_forget()
                    sentbox.pack_forget()
                else:
                    control_panel.pack_forget()
                    viewer.pack_forget()

                    headbox.pack(fill = Y, side = LEFT)
                    sentbox.pack(fill = BOTH, side = LEFT, expand = YES)
                    control_panel.pack(fill = Y, side = LEFT)
                    viewer.pack(fill = BOTH, expand = YES)

            self.pack(fill = BOTH, expand = YES)

        def _change_title(self, prefix, epoch):
            title = [self._fpath_heads[0].folder, prefix]
            bid, _, _, sid = self._selected
            key = f'{bid}_{epoch}'
            if key in self._rpt.sent:
                scores = self._rpt.sent[key][sid]
                if len(scores) == 4:
                    scores = '  '.join(i+f'({j:.2f})' for i, j in zip(('5C@R', 'N-P@R', '5C', 'N-P'), scores))
                else:
                    scores = tuple(scores[i] for i in (1, 3, 4, 11))
                    scores = '  '.join(i+f'({str(j)})' for i, j in zip(('len.', 'P.', 'R.', 'tag.'), scores))
                # else:
                #     r = self._fpath_heads[0].join(f'data.{key}.rpt')
                #     scores = self._rpt.sent[r[5:-4]] = rpt_summary(r, sents = True)
                    
                title.append(scores)
            self.master.title(' | '.join(title))

        def read_listbox(self, event):
            headbox, sentbox = self._boxes
            choice_t = event.widget.curselection()
            if choice_t:
                fpath, heads = self._fpath_heads
                i = int(choice_t[0]) # head/inter-batch id or sentence/intra-batch id
                if event.widget is headbox:
                    head = fpath.join(heads[i])
                    bid, num_word = (int(i) for i in heads[i][5:-4].split('_'))
                    head = pickle_load(head)
                    if head.tag is None:
                        neg_set = set(self._vocabs[0].label.index(i) for i in '01')
                        pos_set = set(self._vocabs[0].label.index(i) for i in '34')
                            
                    sentbox.delete(0, END)
                    for sid, (offset, length, words) in enumerate(zip(head.offset, head.length, head.word)):
                        if head.tag is None:
                            negation = any(i in head.label[sid] for i in pos_set) and any(i in head.label[sid] for i in neg_set)
                        else:
                            negation = False
                        mark = '*' if negation else ''
                        mark += f'{sid}'
                        # if warning_cnt:
                        #     mark += " ◌•▴⨯"[warning_level(warning_cnt)]
                        mark += '\t'
                        sentbox.insert(END, mark + ' '.join(self._vocabs.word[idx] for idx in words[offset:offset + length]))
                    self._selected = bid, num_word, head

                    prefix, suffix = f'data.{bid}_', '.pkl'
                    for fname_time in fpath.listdir():
                        if fname_time.startswith(prefix) and fname_time.endswith(suffix):
                            if fname_time not in self._sent_cache:
                                columns = pickle_load(fpath.join(fname_time))
                                sample_gen = (inf_none_gen if x is None else x for x in columns[:-1])
                                batch = []
                                for i, sample in enumerate(zip(*sample_gen)):
                                    sample = IOData(*sample, inf_none_gen)
                                    stat = SentenceEnergy(sample.mpc_word, sample.mpc_phrase, num_word, sample.offset, sample.length)
                                    batch.append((sample, stat))
                                self._sent_cache[fname_time] = batch, columns[-1]

                elif event.widget is sentbox:
                    bid, num_word, head = self._selected[:3]
                    self._selected = bid, num_word, head, i
                    self.__update_viewer()
            else:
                print('nothing', choice_t, event)

        def _change_bool(self):
            if self._viewer.ready():
                self._viewer.minor_change(self.dynamic_settings(), self.static_settings(0, 1, 2, 3))

        def _change_string(self, event):
            if any(not hasattr(self, attr) for attr in '_entries _panels _checkboxes_entries _checkboxes'.split()):
                return
            if event is None or event.char in ('\r', '\uf700', '\uf701'):
                ss = self.static_settings()
                ss_changed = NumbList(*(t != l for t, l in zip(ss, self._last_numbs)))
                if any(ss_changed[4:]): # font and PCs will not cause resize
                    self.__update_viewer(force_resize = True)
                elif self._viewer.ready():
                    self._viewer.minor_change(self.dynamic_settings(), ss[:4])
                self._last_numbs = ss
                if event and event.char == '\r':
                    self._control_panel.focus()
                
        def shortcuts(self, key_press_event):
            char = key_press_event.char
            # self.winfo_toplevel().bind(key, self._navi)
            if char == 'w':
                self._checkboxes.statistics.ckb.invoke()
                sub_state = 'normal' if self._last_bools.statistics else 'disable'
                self._checkboxes_entries.gauss.ckb.configure(state = sub_state)
                self._checkboxes_entries.gauss.etr.configure(state = sub_state)
            elif char == 'n':
                self._checkboxes.show_nil.ckb.invoke()
            elif char == 'b':
                self._checkboxes.show_paddings.ckb.invoke()
            elif char == 'e':
                self._checkboxes.show_errors.ckb.invoke()
            elif char == '|':
                self._checkboxes.align_coord.ckb.invoke()
            # elif char == 'v':
            #     self._checkboxes.hard_split.ckb.invoke()
            #     if self._last_bools.apply_dash:
            #         self._checkboxes.apply_dash.ckb.invoke()
            elif char == '.':
                self._checkboxes_entries.dash.ckb.invoke()
                # if self._last_bools.hard_split: # turn uni dire off
                #     self._checkboxes.hard_split.ckb.invoke()
            elif char == '\r':
                self._checkboxes_entries.curve.etr.icursor("end")
                self._checkboxes_entries.curve.etr.focus()
            elif char == 'i':
                self._checkboxes.dark_background.ckb.invoke()
                self._checkboxes.inverse_brightness.ckb.invoke()
            elif char == 'u':
                self._checkboxes_entries.curve.ckb.invoke()
            elif char == 'p':
                self._checkboxes_entries.picker.ckb.invoke()
            elif char == 'g':
                self._checkboxes_entries.gauss.ckb.invoke()
            elif char == 'l':
                self._checkboxes_entries.spotlight.ckb.invoke()
            elif char == 'q':
                self._panels.hide_listboxes.ckb.invoke()
            elif char == 'z':
                self._panel_btns[0].invoke()
            elif char == 'x':
                self._panel_btns[1].invoke()
            elif char == 'c':
                self._panel_btns[2].invoke()
            elif char == 'a':
                self._navi('⇤')
            elif char == 's':
                self._navi('↑')
            elif char == ' ':
                self._navi('o')
            elif char == 'd':
                self._navi('↓')
            elif char == 'f':
                self._navi('⇥')
            else:
                print('???', key_press_event)

        def __update_viewer(self, force_resize = False):
            panel_bools = get_checkbox(self._panels)
            changed = (t^l for t, l in zip(panel_bools, self._last_panel_bools))
            changed = PanelList(*changed)
            self._last_panel_bools = panel_bools
            
            viewer = self._viewer
            self.__update_layout(changed.hide_listboxes, changed.detach_viewer or not viewer.winfo_exists())
            viewer = self._viewer

            if len(self._selected) < 4:
                print('selected len:', len(self._selected))
                return

            bid, num_word, head, sid = self._selected
            prefix = f"data.{bid}_"
            suffix = '.pkl'
            timeline = []
            for fname in self._sent_cache:
                if fname.startswith(prefix):
                    timeline.append(fname)

            timeline.sort(key = lambda kv: float(kv[len(prefix):-len(suffix)]))

            num_time = len(timeline)
            if force_resize or not self._viewer.ready(num_word, num_time):
                ds = self.dynamic_settings()
                ss = self.static_settings()
                viewer.configure(num_word, num_time, ds, ss, self._init_time)
                
            viewer.set_framework(head, {t:self._sent_cache[t] for t in timeline})
            viewer.show_sentence(sid)
            viewer.update() # manually update canvas

        def dynamic_settings(self):
            cs = (n for b,n in self._last_combs)
            ns = get_entry(self._checkboxes_entries, comb_types, (n for b,n in self._last_combs), 1)
            self._last_combs = CombList(*zip(cs, ns))

            bs = get_checkbox(self._checkboxes)
            ds = bs + get_checkbox(self._checkboxes_entries, 1) + ns

            # changed_bs = (t^l for t, l in zip(bs, self._last_bools))
            # changed_bs = BoolList(*changed_bs)
            self._last_bools = bs

            # if bs.show_errors and not bs.show_nil:
            #     if changed_bs.show_errors:
            #         self._checkboxes.show_nil.ckb.invoke()
            #     elif changed_bs.show_nil:
            #         self._checkboxes.show_errors.ckb.invoke()
            return DynamicSettings(*ds)

        def static_settings(self, *ids):
            if ids:
                enties = tuple(self._entries[i]    for i in ids)
                types  = tuple(numb_types[i]       for i in ids)
                lasts  = tuple(self._last_numbs[i] for i in ids)
                return get_entry(enties, types, lasts)
            return get_entry(self._entries, numb_types, self._last_numbs)

        def _show_one_tree(self):
            if self._viewer.ready():
                self._viewer.show_tree(False)#, self._viewer, self._viewer._boards[0])

        def _show_all_trees(self):
            if self._viewer.ready():
                self._viewer.show_tree(True)

        def _navi(self, event):
            if self._viewer.ready():
                if isinstance(event, str):
                    self._viewer.navi_to(event)
                elif event.widget in self._navi_btns:
                    self._viewer.navi_to(navi_directions[self._navi_btns.index(event.widget)])

        def _save_canvas(self):
            if not self._viewer.ready():
                return
            bid, _, _, i = self._selected
            options = dict(filetypes   = [('postscript', '.eps')],
                           initialfile = f'{bid}-{i}-{self._viewer.time}.eps',
                           parent      = self)

            filename = filedialog.asksaveasfilename(**options)
            if filename:
                self._viewer.save(filename)

        def _calc_batch(self):
            pass

    def to_circle(xy, xy_orig = None):
        if xy_orig is not None:
            xy = xy - xy_orig
        x, y = (xy[:, i] for i in (0,1))
        reg_angle = np.arctan2(y, x) / np.pi
        reg_angle += 1
        reg_angle /= 2
        reg_power = np.sqrt(np.sum(xy ** 2, axis = 1))
        return np.stack([reg_angle, reg_power], axis = 1), np.max(reg_power)

    def filter_data_coord(x, offset_length, filtered):
        if offset_length is None:
            coord = range(x.shape[0])
        else:
            offset, length = offset_length
            end = offset + length
            coord = range(offset, end)
            x = x[offset:end]
            if filtered is not None:
                filtered = filtered[offset:end]

        if filtered is not None:
            x = x[filtered]
            coord = (c for c, f in zip(coord, filtered) if f)

        return x, coord

    class LayerMPCStat:
        def __init__(self, mpc):
            self._global_data = mpc
            self._xy_dims     = None

        def _without_paddings(self, offset_length):
            return filter_data_coord(self._global_data, offset_length, None)

        def histo_data(self, num_backle, global_xmax = None, offset_length = None, filtered = None):
            x, coord = filter_data_coord(self._global_data[:, 0], offset_length, filtered)
            if global_xmax is None:
                xmin = np.min(x)
                xmax = np.max(x)
            else:
                xmin = 0
                xmax = global_xmax
            if xmin == xmax:
                xmin -= 0.1
                xmax += 0.1

            x = x - xmin # new memory, not in-place
            x /= xmax - xmin # in range [0, 1]
            x //= (1 / num_backle) # bin_width is float, even for floordiv
            collapsed_x = Counter(x)
            cnt_max     = collapsed_x.most_common(1)[0][1]
            x /= num_backle # break cnt into [0, 1] for x coord
            coord       = zip(coord, x)
            xy          = tuple((bint/num_backle, cnt/cnt_max) for bint, cnt in collapsed_x.items())
            return xy, xmin, xmax, coord

        def scatter_data(self, xy_min_max = None, offset_length = None, filtered = None):
            xy, coord = filter_data_coord(self._global_data[:, self._xy_dims], offset_length, filtered)
            if xy_min_max is None:
                xmin = np.min(xy, 0)
                xmax = np.max(xy, 0)
            else:
                xmin, xmax = xy_min_max

            g_orig = np.zeros_like(xmin)
            invalid = g_orig < xmin
            xmin = np.where(invalid, g_orig, xmin)
            invalid = g_orig > xmax
            xmax = np.where(invalid, g_orig, xmax)

            if np.all(xmin == xmax):
                xmin -= 0.1
                xmax += 0.1

            xy = xy - xmin
            xy /= xmax - xmin
            g_orig -= xmin
            g_orig /= xmax - xmin

            return xy, xmin, xmax, g_orig, coord
            # print('xy', xy.shape[0])
            # print('seq_len', seq_len if seq_len else 'None')
            # print('filtered', f'{sum(filtered)} in {len(filtered)}' if filtered else 'None')
            # print('coord', coord)

        def colorify(self, m_max, xy_dims):
            self._xy_dims = xy_dims
            self._global_m_max = m_max
            ene = np.expand_dims(self._global_data[:, 0], 1) / m_max
            ori, sature_max = to_circle(self._global_data[:, xy_dims])
            self._render = np.concatenate([ene, ori], axis = 1)
            return sature_max

        def seal(self, sature_max):
            self._render[:, 2] /= sature_max
            return self

        def __getitem__(self, idx):
            return self._render[idx]

        def __len__(self):
            return self._global_data.shape[0]

    class SentenceEnergy:
        def __init__(self, mpc_word, mpc_phrase, size, offset, length):
            mpc_all = mpc_phrase
            stats = phrase = tuple(LayerMPCStat(l) for l in triangle_to_layers(mpc_phrase))
            if mpc_word is not None:
                mpc_all = np.concatenate([mpc_phrase, mpc_word])
                word = LayerMPCStat(mpc_word)
                stats = (word,) + phrase
            else:
                word = None

            self._min_all = xmax = np.min(mpc_all, 0)
            self._max_all = xmin = np.max(mpc_all, 0)
            for i, stat in enumerate(stats):
                if i == 0:
                    lo = offset, length
                else:
                    _len = i + length - size # pitari without +1
                    if _len <= 0:
                        continue
                    lo = offset, _len
                x, _ = stat._without_paddings(lo)
                _max = np.max(x, 0)
                x = x[x[:, 0] > 0]
                _min = np.min(x, 0)
                invalid = _min < xmin
                xmin = np.where(invalid, _min, xmin)
                invalid = _max > xmax
                xmax = np.where(invalid, _max, xmax)
            self._min_val = xmin
            self._max_val = xmax

            self._stats  = stats
            self._word   = word
            self._phrase = phrase
            self._xy_dim = None
            self._cache  = {}

        def histo_max(self, with_padding):
            if with_padding:
                return self._max_all[0]
            return self._max_val[0]

        def scatter_min_max(self, with_nil, with_padding):
            if with_nil:
                _min = self._min_all[self._xy_dim]
            else:
                _min = self._min_val[self._xy_dim]
            if with_padding:
                _max = self._max_all[self._xy_dim]
            else:
                _max = self._max_val[self._xy_dim]
            return _min, _max

        def make(self, show_paddings, *xy_dims):
            key = xy_dims, show_paddings
            self._xy_dim = xy_dims = np.asarray(xy_dims)
            if key in self._cache:
                self._word, self._phrase = self._cache[key]
                return

            _max = self._max_all if show_paddings else self._max_val
            sature_max = 0
            for stat in self._stats :
                sature_max = max(sature_max, stat.colorify(_max[0], xy_dims))

            if self._word:
                self._word = self._word.seal(sature_max)
            self._phrase = tuple(s.seal(sature_max) for s in self._phrase)
            self._cache[key] = self._word, self._phrase

        @property
        def word(self):
            if self._word:
                return self._word
            return self._phrase[-1]

        @property
        def tag(self):
            return self._phrase[-1]

        @property
        def phrase(self):
            return self._phrase

    def disp(x):
        if x >= 1:
            return str(x)[:3]
        if x <= -1:
            return str(x)[:4]
        if x >= 0.01:
            return str(x)[1:4]
        if x <= -0.01:
            return '-' + str(x)[2:5]
        if x == 0:
            return '0'
        return '+0' if x > 0 else '-0'

    def make_color(mutable_x,
                   show_color = False,
                   inverse_brightness = False,
                   curve = None,
                   fallback_color = 'orange'):
        if show_color:
            v, h, s = mutable_x[:3]
        else:
            if isinstance(mutable_x, float):
                v = mutable_x
            else:
                v = mutable_x[0] if mutable_x.shape else mutable_x
            h = s = 0
        if curve is not None:
            v = curve(v)
        if v < 0 or v > 1:
            return fallback_color
        if inverse_brightness:
            v = 1 - v
        def channel(c):
            x = hex(int(c * 255))
            n = len(x)
            if c < 0:
                return 'z'
            if n == 3:
                return '0' + x[-1]
            elif n == 4:
                return x[2:]
        return '#' + ''.join(channel(x) for x in hsv_to_rgb(h, s, v))

    def make_histogram(stat_board, offx, offy, width, height,
                       stat, offset_length, histo_max,
                       half_word_height,
                       stat_font,
                       stat_color,
                       xlab,
                       filtered = None,
                       distance = None):
        xy, xmin, xmax, coord_x = stat.histo_data(width, histo_max, offset_length, filtered)
        if half_word_height and stat_font and xlab:
            stat_board.create_text(offx, offy + height + half_word_height,
                                   text = disp(xmin),
                                   fill = stat_color, anchor = W, font = stat_font)
            stat_board.create_text(offx + width, offy + height + half_word_height,
                                   text = disp(xmax),
                                   fill = stat_color, anchor = E, font = stat_font)
        # base line
        stat_board.create_line(offx,         offy + height,
                               offx + width, offy + height,
                               fill = stat_color)# does not work, width = 0.5)
        if distance and distance > 0:
            m = 0
            z = sqrt(2* pi* distance)
            a = (2 * distance ** 2)
            data = []
            for i in range(width):
                x = i / width
                y = sum(yj / z * exp(-((x - xj) ** 2 ) / a) for xj, yj in xy)
                if y > m: m = y
                data.append((i, y))
            for i, y in data:
                y /= m
                stat_board.create_line(offx + i, offy + height - y * height,
                                       offx + i, offy + height, fill = stat_color)
        else:
            for x, y in xy:
                stat_board.create_line(offx + x * width, offy + height - y * height,
                                       offx + x * width, offy + height, fill = stat_color)
        return {c:x * width for c, x in coord_x}

    def make_scatter(stat_board, offx, offy, width, height, r,
                     stat, offset_length, scatter_min_max,
                     stat_color,
                     half_word_height,
                     stat_font,
                     to_color,
                     background,
                     xlab, ylab, clab,
                     filtered = None):
        # globals().update({k:v for k,v in zip(BoardConfig._fields, config)})
        # stat_board.create_line(offx, offy, offx, offy + height, fill = stat_color)
        # stat_board.create_line(offx, offy, offx + width,  offy, fill = stat_color)
        # stat_board.create_line(offx, offy + height, offx + width, offy + height, fill = stat_color)
        # stat_board.create_line(offx + width,  offy, offx + width, offy + height, fill = stat_color)
        # if distance and distance > 0:
        #     for i in range(width):
        #         for j in range(int(height)):
        #             x = i / width
        #             y = j / height
        #             z = sum(exp(-((x - xj) ** 2 )/ (2*distance**2)-((y - yj) ** 2 )/ (2*distance**2)) for xj, yj in xy) / len(xy)
        #             stat_board.create_line(offx + i,     offy + height - j,
        #                                    offx + i + 1, offy + height - j, fill = make_color(z))

        xy, xy_min, xy_max, xy_orig, coord = stat.scatter_data(scatter_min_max, offset_length, filtered)
            
        if half_word_height and stat_font and (xlab or ylab or clab):
            xmin, ymin = (disp(i) for i in xy_min)
            xmax, ymax = (disp(i) for i in xy_max)
            if xlab:
                stat_board.create_text(offx, offy + height + half_word_height,
                                        text = xmin, fill = stat_color,
                                        font = stat_font, anchor = W)
                stat_board.create_text(offx + width, offy + height + half_word_height,
                                        text = xmax, fill = stat_color,
                                        font = stat_font, anchor = E)
            if ylab:
                stat_board.create_text(offx - half_word_height * 1.3, offy,
                                        text = ymax, fill = stat_color,
                                        font = stat_font)#, angle = 90)
                stat_board.create_text(offx - half_word_height * 1.3, offy + height,
                                        text = ymin, fill = stat_color,
                                        font = stat_font)
            if clab:
                stat_board.create_text(offx - half_word_height * 1.3, offy + height / 2,
                                        text = f'{len(xy)}•', fill = 'SpringGreen3',
                                        font = stat_font)#, anchor = E)

        stat_board.create_rectangle(offx,         offy,
                                    offx + width, offy + height,
                                    fill = background, outline = background)
        x, y = xy_orig
        lx = offx + x * width
        ly = offy + height - y * height
        stat_board.create_line(lx - 4, ly,
                               lx + 4, ly, fill = 'red')
        stat_board.create_line(lx, ly - 4,
                               lx, ly + 4, fill = 'red')

        scatter_coord_item = {}
        for c, (x, y) in zip(coord, xy):
            x *= width
            y *= height
            item = stat_board.create_oval(offx + x - r, offy + height - y - r,
                                          offx + x + r, offy + height - y + r,
                                          fill = to_color(stat[c]), outline = '')
            scatter_coord_item[c] = item
        return scatter_coord_item

    # NumbList: 'word_width, word_height, xy_ratio, histo_width, scatter_width'
    #  effect:      b             s,b   <-   s,b        s               s
    # dark_background, inverse_brightness, delta_shape, show_errors, statistics, show_paddings, show_nil, align_coord, show_color
    #        sb                 b              sb            b              s?              sb            s           s            b
    fields = ', num_word, half_word_width, half_word_height, line_dx, line_dy, deco_dx, deco_dy, upper_padding, canvas_width, canvas_height, stat_font, stat_pad_left, stat_pad_between, stat_pad_right'
    FrameGeometry = namedtuple('FrameGeometry', ','.join(NumbList._fields) + fields)
    BoardConfig = namedtuple('BoardConfig', DynamicSettings._fields + FrameGeometry._fields)

    class ThreadViewer(Frame):
        def __init__(self,
                     master,
                     vocabs,
                     time_change_callback):
            super().__init__(master)
            self._time_change_callback = time_change_callback
            self._boards = Canvas(self), Canvas(self)
            self._time_slider = None, Scale(self, command = self._change_time)
            self._vocabs = vocabs
            self._spotlight_subjects = None
            
        def configure(self,
                      num_word,
                      num_time,
                      dynamic_settings,
                      static_settings,
                      init_time):
            # calculate canvas
            half_word_width  = static_settings.word_width >> 1
            half_word_height = static_settings.word_height >> 1
            line_dx = half_word_width - static_settings.word_height / static_settings.xy_ratio
            line_dy = line_dx * static_settings.xy_ratio
            deco_dx = static_settings.line_width / np.sqrt(1 + static_settings.xy_ratio ** 2) / 2
            deco_dy = deco_dx * static_settings.xy_ratio
            upper_padding = max(half_word_height, static_settings.line_width / 2)
            canvas_width  = num_word * static_settings.word_width
            canvas_height = (num_word + 2) * (static_settings.word_height + line_dy) + upper_padding # +2 for word and tag layer
            stat_paddings = (28, 10, 22)
            bcfg = num_word, half_word_width, half_word_height, line_dx, line_dy, deco_dx, deco_dy, upper_padding, canvas_width, canvas_height, ('helvetica', 10)
            self._frame_geometry = FrameGeometry(*(static_settings + bcfg + stat_paddings))
            self._conf = BoardConfig(*(dynamic_settings + self._frame_geometry))
            self._last_show_paddings = dynamic_settings.show_paddings

            # resize canvas
            board, stat_board = self._boards
            board.config(width  = canvas_width,
                         height = canvas_height,
                         bd = 0, highlightthickness = 0, # cancel white label_line_bo
                         cursor = 'fleur',
                         scrollregion = '0 0 %d %d' % (canvas_width, canvas_height))

            stat_width = sum(stat_paddings) + static_settings.histo_width + static_settings.scatter_width
            stat_board.config(width  = stat_width,
                              height = canvas_height,
                              bd = 0, highlightthickness = 0,
                              scrollregion = '0 0 %d %d' % (stat_width, canvas_height))

            def scroll_start(event):
                board.scan_mark(event.x, event.y)
                stat_board.scan_mark( 0, event.y)
                self.focus()
            def scroll_move(event):
                board.scan_dragto(event.x, event.y, gain = 1)
                stat_board.scan_dragto( 0, event.y, gain = 1)
            board.bind("<ButtonPress-1>", scroll_start)
            board.bind("<B1-Motion>",     scroll_move)
            # def scroll_delta(event):
            #     print(event)
                # board.xview_scroll(-1 * int(event.delta / 60), "units")
            # board.bind("<MouseWheel>", scroll_delta)
            def moved(event):
                x = board.canvasx(event.x)
                y = board.canvasy(event.y)
                self.spotlight(x, y)
                # print(x, y, board.find_closest(x, y))
            board.bind("<Motion>", moved)

            last_num_time, time_slider = self._time_slider
            time_slider.config(to = num_time - 1, tickinterval = 1, showvalue = False)
            time_slider.set(init_time)
            self._time_slider = num_time, time_slider
            self._animation = []

            # initial view position
            if self._conf.delta_shape:
                self._boards[0].yview_moveto(1)
                self._boards[1].yview_moveto(1)
            else:
                self._boards[0].yview_moveto(0)
                self._boards[1].yview_moveto(0)
            self._refresh_layout()
            # hscroll = Scrollbar(self, orient = HORIZONTAL)
            # vscroll = Scrollbar(self, orient = VERTICAL)
            # self._hv_scrolls = None, vscroll
            # hscroll.config(command = board.xview)
            # vscroll.config(command = board.yview)
            # hscroll.pack(fill = X, side = BOTTOM)
            # vscroll.pack(fill = Y, side = RIGHT)
            # xscrollcommand = self._hv_scrolls[0].set,
            # yscrollcommand = self._hv_scrolls[1].set,
            # need support for 2-finger gesture
            # board.bind("<Button-6>", scroll_start)
            # board.bind("<Button-7>", scroll_move)

        # @property funny mistake
        def ready(self, num_word = None, num_time = None):
            last_num_time = self._time_slider[0]
            if last_num_time is None:
                return False
            elif num_time is not None and last_num_time != num_time:
                return False
            if num_word is not None and self._conf.num_word != num_word:
                return False
            return True

        @property
        def time(self):
            if self._time_slider[0] is None:
                return 0
            return self._time_slider[1].get()

        def set_framework(self, head, time_data):
            if len(time_data) != self._time_slider[0]:
                raise ValueError(f'Set Invalid timesteps! {len(time_data)} vs. {self._time_slider[0]}')
            self._head_time = head, tuple(time_data.items()) # [(fname, sents), ..]

        def show_sentence(self, sid): # across timesteps
            head, time_data = self._head_time
            self._head = IOHead(*tuple(None if h is None else h[sid] for h in head))
            self._data = []
            for _, (batch, _) in time_data:
                self._data.append(batch[sid])
            self._refresh_board(self._time_slider[1].get())
            if not self._conf.show_paddings:
                self._boards[0].xview_moveto(self._head.offset / self._conf.num_word)

        def minor_change(self, dynamic_settings, dynamic_geometry):
            self._conf = BoardConfig(*(dynamic_settings + dynamic_geometry + self._frame_geometry[len(dynamic_geometry):]))
            if dynamic_settings.statistics ^ self._last_show_paddings:
                self._refresh_layout()
                self._last_show_paddings = dynamic_settings.statistics
            self._refresh_board(self._time_slider[1].get())

        def _change_time(self, time):
            self._refresh_board(int(time))

        def _refresh_layout(self):
            # dynamic layout
            for w in (self._time_slider + self._boards)[1:]:
                w.pack_forget()
            if self._time_slider[0] > 1:
                self._time_slider[1].pack(fill = Y, side = LEFT)
            if self._conf.statistics:
                self._boards[1].pack(side = LEFT, expand = YES)
                self._boards[0].pack(side = LEFT, expand = YES)
            else:
                self._boards[0].pack(side = LEFT, expand = YES)

        def _refresh_board(self, tid):
            board, stat_board = self._boards
            board.delete('elems')
            stat_board.delete(ALL)

            fg_color = 'black'
            bg_color = 'white'
            stat_fg = 'gray5'
            stat_bg = 'gray94'
            if self._conf.dark_background:
                fg_color, bg_color = bg_color, fg_color
                stat_fg = 'gray40'
                stat_bg = 'gray10'
            to_color = partial(make_color,
                               show_color         = self._conf.show_color and not self._conf.apply_dash,
                               inverse_brightness = self._conf.inverse_brightness,
                               curve              = self._conf.curve if self._conf.apply_curve else None) #self._curve if apply_curve else
            for b in self._boards:
                b.configure(background = bg_color)

            data, stat = self._data[tid]
            stat.make(self._conf.show_paddings, self._conf.pc_x, self._conf.pc_y)
            label_layers = self.__draw_board(data, stat, fg_color, to_color)
            if self._conf.statistics:
                self.__draw_stat_board(label_layers, data.offset, data.length, stat, stat_fg, stat_bg, to_color)
            
            _, time = self._head_time
            bid, epoch = time[tid][0][5:-4].split('_')
            scores = '  '.join(i+f'({data.scores[j]})' for i, j in zip(('len.', 'P.', 'R.', 'tag.'), (1, 3, 4, 11)))
            title = f'Batch: {bid} Epoch: {epoch} ' + scores
            self._time_change_callback(title, epoch) # [(fname, data)]

        def navi_to(self, navi_char, steps = 0, num_frame = 24, duration = 1):
            self.clear_animation()
            board, stat_board = self._boards
                
            num_word = self._conf.num_word
            show_paddings = self._conf.show_paddings
            head = self._head

            xpad = board.xview() # %
            ypad = board.yview() # %
            ox, oy = xpad[0], ypad[0]
            xpad = xpad[1] - xpad[0]
            ypad = ypad[1] - ypad[0]
            l_pad = (head.offset / num_word)
            ratio = 1 - l_pad * (num_word - head.length)
            r_pad = ratio - xpad
            c_pad = 0.5 - xpad / 2 if show_paddings else (l_pad + r_pad) / 2
            o_pad = ratio / 2

            if self._conf.delta_shape:
                navi_left   = (0        if show_paddings else l_pad, 1 - ypad)
                navi_right  = (1 - xpad if show_paddings else r_pad, 1 - ypad)
                navi_top    = (c_pad, 0 if show_paddings else 1 - ratio)
                navi_center = (c_pad, 0.5 - ypad / 2 if show_paddings else 1 - o_pad - ypad / 2)
                navi_bottom = (c_pad, 1 - ypad)
            else:
                navi_left   = (0        if show_paddings else l_pad, 0)
                navi_right  = (1 - xpad if show_paddings else r_pad, 0)
                navi_top    = (c_pad, 0)
                navi_center = (c_pad, 0.5 - ypad / 2 if show_paddings else o_pad - ypad / 2)
                navi_bottom = (c_pad, 1 - ypad if show_paddings else ratio - ypad)

            if steps > 0:
                clip = lambda l,x,u: l if x < l else (u if x > u else x)
                if navi_char in '⇤⇥':
                    dx = steps * (self._conf.word_width)
                    if navi_char == '⇤':
                        dx = -dx
                    dx /= self._conf.canvas_width # %
                    dy = 0
                elif navi_char in '↑↓':
                    dx = 0
                    dy = steps * (self._conf.word_height + self._conf.line_dy)
                    if navi_char == '↑':
                        dy = -dy
                    dy /= self._conf.canvas_height # %
                nx = clip(navi_left[0], ox + dx, navi_right[0])
                ny = clip(navi_top[1], oy + dy, navi_bottom[1])
            else:
                nx, ny = {'⇤': navi_left, '⇥': navi_right, '↑': navi_top, '↓': navi_bottom, 'o': navi_center}[navi_char]

            if nx == ox and ny == oy:
                return
                
            def smoothen(ox, nx):
                x = np.linspace(ox, nx, num_frame)
                if ox == nx:
                    return x
                x_mean = (ox + nx) * 0.5
                x_diff = (nx - ox) * 0.5
                x -= x_mean
                x /= x_diff
                np.sin(x * np.pi / 2, out = x)
                x *= x_diff
                x += x_mean
                return x
                
            i = int(duration / num_frame * 1000)
            j = 0
            def move_anime(xi, yi, l):
                board.xview_moveto(xi)
                board.yview_moveto(yi)
                stat_board.yview_moveto(yi)
                l.pop(0)
            for xi, yi in zip(smoothen(ox, nx), smoothen(oy, ny)):
                self._animation.append(board.after(j, move_anime, xi, yi, self._animation))
                j += i

        def clear_animation(self):
            while self._animation:
                a = self._animation.pop()
                self._boards[0].after_cancel(a)
                self._boards[1].after_cancel(a)

        def save(self, fname):
            if self._conf.show_paddings:
                x = 0
                y = 0
                w = self._conf.canvas_width
                h = self._conf.canvas_height
            else:
                n = self._conf.num_word - self._head.length + 2
                l = self._conf.word_height + self._conf.line_dy

                x = self._conf.word_width
                w = self._conf.canvas_width - (n * x)
                n -= 1 # for tag layer
                if self._conf.delta_shape:
                    y = n * l
                    h = self._conf.canvas_height - y
                else:
                    y = 0
                    h = self._conf.canvas_height - n * l
            
            self._boards[0].postscript(file = fname, x = x, y = y, width  = w, height = h,
                                       colormode = 'color' if self._conf.show_color else 'gray')

        def spotlight(self, x, y):
            board, stat_board = self._boards
            board_item_coord, w, h, last_picker, last_spotlight, last_bbox = self._spotlight_subjects

            if self._conf.apply_spotlight:
                radius = self._conf.spotlight
                if last_spotlight:
                    nx, ny, _, _ = board.bbox(last_spotlight)
                    board.move(last_spotlight, x - radius - nx, y - radius - ny)
                else:
                    light = make_color(self._conf.picker / 2, inverse_brightness = not self._conf.dark_background)
                    last_spotlight = board.create_oval(x - radius, y - radius, x + radius, y + radius, fill = light, outline = '')
                    self._spotlight_subjects = board_item_coord, w, h, last_picker, last_spotlight, last_bbox
                board.tag_lower(last_spotlight)
            elif last_spotlight:
                board.delete(last_spotlight)
                last_spotlight = None

            if not self._conf.apply_picker:
                board.delete(last_picker)
                self._spotlight_subjects = board_item_coord, w, h, None, last_spotlight, last_bbox
                return

            bbox = None
            for bbox in board_item_coord:
                l, t = bbox
                if l < x < l + w and t < y < t + h:
                    if bbox == last_bbox:
                        # print('avoid repainting')
                        return
                    elif last_picker is not None:
                        # print('repaint')
                        board.delete(last_picker)
                    light = make_color(self._conf.picker, inverse_brightness = not self._conf.dark_background)
                    act = board.create_rectangle(*bbox, l + w, t + h, fill = light, outline = '')
                    board.tag_lower(act)
                    _, coord = board_item_coord[bbox]
                    # for item in board_item:
                    #     board.tag_raise(item)
                    self._spotlight_subjects = board_item_coord, w, h, act, last_spotlight, bbox
                    if self._conf.statistics:
                        items, positions, reflection = self._spotlight_objects
                        for i in reflection:
                            stat_board.delete(i)
                        x,y,x_,y_ = stat_board.bbox(items[coord])
                        stat_ref = stat_board.create_oval(x-1,y-1,x_+1,y_+1, fill = 'red', outline = '')
                        x,y,h = positions[coord]
                        histo_ref = stat_board.create_line(x,y,x,y+h, fill = 'red')
                        self._spotlight_objects = items, positions, (stat_ref, histo_ref)
                    break
                else:
                    bbox = None
            if bbox is None and last_picker is not None:
                # print('remove spotlight when nothing found')
                board.delete(last_picker)
                self._spotlight_subjects = self._spotlight_subjects[:-3] + (None, last_spotlight, None)
                if self._conf.statistics:
                    items, positions, reflection = self._spotlight_objects
                    for i in reflection:
                        stat_board.delete(i)
                    self._spotlight_objects = items, positions, tuple()

        def show_tree(self, show_all_trees, *frame_canvas):
            label = Tree.fromstring(self._head.tree)
            label.set_label(label.label() + ' (corpus)')

            if frame_canvas:
                frame, canvas = frame_canvas
            else:
                frame  = CanvasFrame()
                canvas = frame.canvas()
            label = TreeWidget(canvas, label, draggable = 1) # bbox is not sure
            label.bind_click_trees(label.toggle_collapsed)
            frame.add_widget(label)
            pad = 20
            inc = 0
            below = label.bbox()[3] + pad
            right_bound = canvas.winfo_screenwidth() # pixel w.r.t hidpi (2x on this mac)

            def at_time(pred, left_top, force_place = False):
                pred = TreeWidget(canvas, pred,  draggable = 1)
                pred_wh = pred.bbox()
                pred_wh = pred_wh[2] - pred_wh[0], pred_wh[3] - pred_wh[1]
                if not force_place and pred_wh[0] + left_top[0] > right_bound:
                    left_top = 0, below + inc
                pred.bind_click_trees(pred.toggle_collapsed)
                frame.add_widget(pred, *left_top)
                return left_top + pred_wh

            left_top = (0, below)
            if show_all_trees:
                tids = []
                trees = []
                for tid in range(self._time_slider[0]):
                    data, _ = self._data[tid]
                    tree = Tree.fromstring(data.tree)
                    if tree in trees:
                        tids[trees.index(tree)].append(tid)
                    else:
                        trees.append(tree)
                        tids.append([tid])
                for ts, pred in zip(tids, trees):
                    pred.set_label(pred.label() + ' (predicted-%s)' % ','.join(str(tid) for tid in ts))
                    ltwh = at_time(pred, left_top, force_place = left_top[0] == 0)
                    inc = max(inc, ltwh[3])
                    if ltwh[0] != left_top[0]: # new line
                        below += inc + pad
                    left_top = (ltwh[0] + ltwh[2] + pad, below)
            else:
                tid = self._time_slider[1].get()
                data, _ = self._data[tid]
                pred = Tree.fromstring(data.tree)
                pred.set_label(pred.label() + ' (predicted-%d)' % tid)
                at_time(pred, left_top, force_place = True)

        def __draw_stat_board(self, label_layers, offset, length, stat, fg_color, bg_color, to_color):
            stat_board = self._boards[1]
            vocabs = self._vocabs
            nil = vocabs.label.index(NIL) if vocabs.tag else -1
            scatter_coord_item = {}
            histo_coord_position = {}
            # half_word_height, word_height, line_dy, delta_shape, canvas_height, show_paddings, show_nil   | histo_width | scatter_width | -> stat_width (pad_left, )

            line_dy = self._conf.line_dy
            incre_y = line_dy + self._conf.word_height
            if self._conf.delta_shape:
                offy = self._conf.canvas_height - incre_y
                incre_y = -incre_y
                bottom_offy = offy + incre_y
            else:
                bottom_offy = offy = self._conf.upper_padding

            scatter_width = self._conf.scatter_width
            histo_width   = self._conf.histo_width
            bottom_height = 2 * line_dy + self._conf.word_height
            pad_left      = self._conf.stat_pad_left
            histo_offset = pad_left + scatter_width + self._conf.stat_pad_between
            def level_tag(offy, tag):
                if self._conf.delta_shape:
                    offy = offy - incre_y - self._conf.half_word_height
                else:
                    offy += self._conf.half_word_height - self._conf.upper_padding
                stat_board.create_text(histo_offset + histo_width + self._conf.stat_pad_right, offy, text = tag, fill = 'deep sky blue', anchor = E)

            _scatter = partial(make_scatter,
                               stat_board       = stat_board,
                               offx             = pad_left,
                               width            = scatter_width,
                               r                = self._conf.line_width / 2,
                               scatter_min_max  = stat.scatter_min_max(self._conf.show_nil, self._conf.show_paddings) if self._conf.align_coord else None,
                               stat_color       = fg_color,
                               stat_font        = self._conf.stat_font,
                               half_word_height = self._conf.half_word_height,
                               background       = bg_color,
                               to_color         = to_color)
            
            _histo = partial(make_histogram,
                             stat_board       = stat_board,
                             offx             = histo_offset,
                             width            = histo_width,
                             histo_max        = stat.histo_max(self._conf.show_paddings) if self._conf.align_coord else None,
                             stat_color       = fg_color,
                             stat_font        = self._conf.stat_font,
                             half_word_height = self._conf.half_word_height,
                             distance = self._conf.gauss if self._conf.apply_gauss else None)
            bottom_offset_length = None if self._conf.show_paddings else (offset, length)
            level_tag(offy, tag = 'W')

            if self._conf.align_coord:
                height = line_dy + self._conf.word_height * 0.75
                xlab = False
            else:
                height = line_dy
                xlab = True
            sci = _scatter(offy = bottom_offy, stat = stat.word, offset_length = bottom_offset_length, height = bottom_height, xlab = True, ylab = True, clab = not xlab).items()
            hcp = _histo  (offy = bottom_offy, stat = stat.word, offset_length = bottom_offset_length, height = bottom_height, xlab = True).items()
            for (i, it), (j, ip) in zip(sci, hcp):
                ip = ip + histo_offset, bottom_offy, bottom_height
                scatter_coord_item  [('w', i)] = it
                histo_coord_position[('w', j)] = ip
            offy += incre_y * 2 # skip .tag
            for l, (plabel_layer, layer_phrase_energy) in enumerate(zip(label_layers, reversed(stat.phrase))): # TODO: not match, historic bug
                if self._conf.show_paddings or l < length: # watch out for not showing and len <= 2
                    cond_level_len  = None if self._conf.show_paddings else (offset, length - l)
                    cond_nil_filter = None if self._conf.show_nil else [s != nil for s in plabel_layer] # interesting! tuple is not a good filter here, list is proper!
                    level_tag(offy, str(l))
                    sci = _scatter(offy = offy, stat = layer_phrase_energy, offset_length = cond_level_len, filtered = cond_nil_filter, height = height, xlab = xlab, ylab = xlab, clab = not xlab).items()
                    hcp = _histo  (offy = offy, stat = layer_phrase_energy, offset_length = cond_level_len, filtered = cond_nil_filter, height = height, xlab = xlab).items()
                    for (i, it), (j, ip) in zip(sci, hcp):
                        scatter_coord_item  [(l, j)] = it
                        histo_coord_position[(l, i)] = ip + histo_offset, offy, height
                offy += incre_y
            self._spotlight_objects = scatter_coord_item, histo_coord_position, tuple()

        def __draw_board(self, data, stat, fg_color, to_color):
            head   = self._head
            board  = self._boards[0]
            vocabs = self._vocabs
            board_item_coord = {}
            apply_dash = self._conf.apply_dash
            tail_ = 1 - self._conf.dash
            tail_ = 1 / tail_

            line_dx = self._conf.line_dx
            line_dy = -self._conf.line_dy
            word_center     = self._conf.half_word_height                   # >--- word ---<
            level_unit      = self._conf.word_height + self._conf.line_dy   # word lines
            tag_label_center  = word_center + level_unit                      # >--- tag label ---<
            tag_label_line_bo = 2 * level_unit                                # word lines tag lines
            w_p_s = self._conf.word_height, level_unit, level_unit + self._conf.word_height # word >--> tag >--> label
            line_width = self._conf.line_width
            r = line_width // 2
            decorate = isinstance(line_width, int) and line_width % 2 == 0
            deco_dx = 0 if decorate else self._conf.deco_dx
            deco_dy = 0 if decorate else self._conf.deco_dy
            font_name, font_size = self._conf.font
            round_int = lambda x: int(round(x))
            if self._conf.delta_shape:
                line_dy = self._conf.line_dy
                deco_dy = - deco_dy
                word_center        = self._conf.canvas_height - word_center
                tag_label_center   = self._conf.canvas_height - tag_label_center
                tag_label_line_bo  = self._conf.canvas_height - tag_label_line_bo
                w_p_s = tuple(self._conf.canvas_height - b for b in w_p_s)
                               
            for i, w in enumerate(head.word):
                if not self._conf.show_paddings and not (head.offset <= i < head.offset + head.length):
                    continue

                center_x = (i + 0.5) * self._conf.word_width
                left_x   = center_x - self._conf.half_word_width
                if self._conf.delta_shape:
                    wbox = (left_x, w_p_s[1]         )
                    pbox = (left_x, tag_label_line_bo)
                else:
                    wbox = (left_x, 0       )
                    pbox = (left_x, w_p_s[1])

                word_color = to_color(1.0 if apply_dash else stat.word[i])
                word = vocabs.word[w]

                if data.tag is None:
                    elems = [board.create_text(center_x, tag_label_center, text = word, font = (font_name, font_size),
                                               fill = word_color, tags = ('elems', 'node'))]
                    elems.append(board.create_line(center_x,  w_p_s[2],
                                                   center_x,  tag_label_line_bo,
                                                   width = line_width,
                                                   fill = word_color,
                                                   tags = ('elems', 'line')))
                    board_item_coord[pbox] = elems, ('p', i)
                else:
                    node = board.create_text(center_x, word_center, text = word, font = (font_name, font_size),
                                             fill = word_color, tags = ('elems', 'node'))
                    line = board.create_line(center_x,  w_p_s[0],
                                             center_x,  w_p_s[1],
                                             width = line_width,
                                             fill = word_color,
                                             tags = ('elems', 'line'))
                    if apply_dash or not decorate:
                        elems = node, line
                    else:
                        sdot = board.create_oval(center_x - r, w_p_s[0] - r,
                                                center_x + r, w_p_s[0] + r,
                                                fill = word_color, outline = '', tags = ('elems', 'dot'))
                        edot = board.create_oval(center_x - r, w_p_s[1] - r,
                                                center_x + r, w_p_s[1] + r,
                                                fill = word_color, outline = '', tags = ('elems', 'dot'))
                        elems = node, line, sdot, edot
                    board_item_coord[wbox] = elems, ('w', i)
                    tp = head.tag[i]
                    pp = data.tag[i]
                    tag_color = to_color(data.tag_score if apply_dash else stat.tag[i])
                    node = board.create_text(center_x, tag_label_center,
                                             fill = tag_color, font = (font_name, font_size if apply_dash else round_int(font_size * data.tag_score[i])),
                                             text = f'{vocabs.tag[pp]}' if not self._conf.show_errors or pp == tp else f'{vocabs.tag[pp]}({vocabs.tag[tp]})',
                                             tags = ('elems', 'node'))
                    line = board.create_line(center_x,  w_p_s[2],
                                             center_x,  tag_label_line_bo,
                                             width = line_width,
                                             fill = tag_color,
                                             tags = ('elems', 'line'))
                    if pp != tp and self._conf.show_errors:
                        err = board.create_rectangle(*board.bbox(node),
                                                     outline = 'red',
                                                     dash = (1, 2),
                                                     tags = ('elems', 'err'))
                        elems = node, line, err
                    else:
                        elems = node, line
                    if not apply_dash and decorate:
                        sdot = board.create_oval(center_x - r, w_p_s[2] - r,
                                                 center_x + r, w_p_s[2] + r,
                                                 fill = tag_color, outline = '', tags = ('elems', 'dot'))
                        edot = board.create_oval(center_x - r, tag_label_line_bo - r,
                                                 center_x + r, tag_label_line_bo + r,
                                                 fill = tag_color, outline = '', tags = ('elems', 'dot'))
                        elems = elems + (sdot, edot)
                    board_item_coord[pbox] = elems, (0, i)

            if vocabs.tag: # in both meanning: label and pol cate
                nil = vocabs.label.index(NIL)
            else:
                nil = -1
            tlabel = triangle_to_layers(head.label)
            plabel = triangle_to_layers(data.label)
            tright = triangle_to_layers(head.right)
            pright = triangle_to_layers(data.right)
            label_score = triangle_to_layers(data.label_score)
            split_score = triangle_to_layers(data.split_score)
            label_score.reverse()
            split_score.reverse()
            tlabel.reverse()
            plabel.reverse()
            tright.reverse()
            pright.reverse()
            for l, layers in enumerate(zip(plabel, tlabel, pright, tright)): # , data.energy.left, data.energy.right
                last_line_bo = tag_label_line_bo
                if self._conf.delta_shape:
                    tag_label_center = tag_label_line_bo - self._conf.half_word_height # >--- label ---<
                    line_y         = tag_label_line_bo - self._conf.word_height
                    tag_label_line_bo -= level_unit
                else:
                    tag_label_center = tag_label_line_bo + self._conf.half_word_height
                    line_y         = tag_label_line_bo + self._conf.word_height
                    tag_label_line_bo += level_unit

                for p, (ps, ts, pr, tr) in enumerate(zip(*layers)):
                    if not self._conf.show_paddings and not (head.offset <= p < head.offset + head.length - l) or not self._conf.show_nil and ps == nil: # not use ts because of spotlight
                        continue
                    center_x = (l/2 + p + .5) * self._conf.word_width
                    left_x   = center_x - self._conf.half_word_width
                    sbox = (left_x, tag_label_line_bo) if self._conf.delta_shape else (left_x, last_line_bo)
                    lid = self._conf.num_word - l - 1
                    mpc_color = to_color(stat.phrase[lid][p])
                    if apply_dash:
                        label_color = to_color(label_score[l][p])
                    else:
                        label_color = mpc_color
                    
                    elems = [board.create_text(center_x, tag_label_center,
                                               text = f'{vocabs.label[ps]}' if  not self._conf.show_errors or ps == ts else f'{vocabs.label[ps]}({vocabs.label[ts]})',
                                               fill = label_color,
                                               tags = ('elems', 'node'), font = (font_name, font_size if apply_dash else round_int(font_size * label_score[l][p])),)]
                    if ps != ts and self._conf.show_errors:
                        elems.append(board.create_rectangle(*board.bbox(elems[0]),
                                                            outline = 'red', dash = (1, 2),
                                                            tags = ('elems', 'err')))
                    board_item_coord[sbox] = elems, (l, p)

                    if not self._conf.show_paddings:
                        if l >= head.length - 1:
                            continue

                    if apply_dash:
                        if pr:
                            score = split_score[l][p]
                            to_x = center_x + line_dx
                        else:
                            score = 1 - split_score[l][p]
                            to_x = center_x - line_dx
                        score -= 0.5
                        score *= 2
                        color = to_color(score)
                        score -= self._conf.dash
                        score *= tail_
                        width = line_width * score
                        if width < 1:
                            width = 1
                        dash_ = None
                        if score < 0:
                            dash_ = int(-tail_ / score)
                            if dash_ > 255:
                                dash_ = 255
                            dash_ = (dash_, 1)

                        elems.append(board.create_line(center_x, line_y, to_x, tag_label_line_bo,
                                                       width = width, fill = color, dash = dash_,
                                                       tags = ('elems', 'line')))
                    else:
                        right_score = split_score[l][p]
                        left_score  = 1 - right_score
                        offset_right_x = line_dx * right_score
                        offset_left_x  = line_dx * left_score
                        to_left_x  = center_x - offset_left_x
                        to_right_x = center_x + offset_right_x
                        to_left_y  = line_y - line_dy * left_score
                        to_right_y = line_y - line_dy * right_score
                        if offset_right_x > 0.5:
                            elems.append(board.create_line(center_x   - deco_dx,     line_y - deco_dy,
                                                           to_right_x + deco_dx, to_right_y + deco_dy,
                                                           width = line_width, fill = mpc_color,
                                                           tags = ('elems', 'r_line')))
                        if offset_left_x > 0.5:
                            elems.append(board.create_line(center_x  + deco_dx,    line_y - deco_dy,
                                                           to_left_x - deco_dx, to_left_y + deco_dy,
                                                           width = line_width, fill = mpc_color,
                                                           tags = ('elems', 'l_line')))
                        if decorate:
                            if offset_right_x > 0.5:
                                elems.append(board.create_oval(to_right_x - r, to_right_y - r,
                                                               to_right_x + r, to_right_y + r,
                                                               fill = mpc_color, outline = '', tags = ('elems', 'r_dot')))
                            if offset_left_x > 0.5:
                                elems.append(board.create_oval(to_left_x - r, to_left_y - r,
                                                               to_left_x + r, to_left_y + r,
                                                               fill = mpc_color, outline = '', tags = ('elems', 'l_dot')))
                            elems.append(board.create_oval(center_x - r, line_y - r,
                                                           center_x + r, line_y + r,
                                                           fill = mpc_color, outline = '', tags = ('elems', 'dot')))

            if self._spotlight_subjects:
                self._spotlight_subjects = (board_item_coord, self._conf.word_width, level_unit) + self._spotlight_subjects[-3:]
            else:
                self._spotlight_subjects = board_item_coord, self._conf.word_width, level_unit, None, None, None
            return plabel

    import argparse
    import getpass
    def get_args():
        parser = argparse.ArgumentParser(
            prog = 'Visual', usage = '%(prog)s DIR [options]',
            description = 'A handy viewer for parsing and its joint experiments', add_help = False
        )
        parser.add_argument('dir', metavar = 'DIR', help = 'indicate a local or remote directory', type = str)
        parser.add_argument('-h', '--host',     help = 'remote host',     type = str, default = 'localhost')
        parser.add_argument('-u', '--username', help = 'remote username', type = str, default = getpass.getuser())
        parser.add_argument('-p', '--port',     help = 'remote port, necessary for remote connection or your tunnel, will ask for password', type = int, default = -1)
        args = parser.parse_args()
        if not isinstance(args.dir, str):
            parser.print_help()
            print('Please provide an folder with -d/--dir', file = sys.stderr)
            exit()
        if args.port > 0 or args.host != 'localhost':
            import pysftp
            import paramiko
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            hostname = args.host
            username = args.username
            port     = args.port
            cfile = join(expanduser('~'), '.ssh', 'config')
            if isfile(cfile):
                user_ssh_config = paramiko.config.SSHConfig()
                with open(cfile) as cfile:
                    user_ssh_config.parse(cfile)
                if args.host in user_ssh_config.get_hostnames():
                    config = user_ssh_config.lookup(args.host)
                    hostname = config['hostname']
                    username = config.get('user', args.username)
                    if port < 0:
                        port = config.get('port', 22)

            password = getpass.getpass('Password for %s:' % username)
            # try:
            sftp = pysftp.Connection(hostname,
                                     username = username,
                                     password = password,
                                     port     = port,
                                     cnopts   = cnopts)
            files = sftp.listdir(args.dir)
            return PathWrapper(args.dir, sftp)
        return PathWrapper(args.dir, None)

    if __name__ == '__main__':
        # root.geometry("300x300+300+300")
        root = Tk()
        app = TreeExplorer(root, get_args())
        root.mainloop()