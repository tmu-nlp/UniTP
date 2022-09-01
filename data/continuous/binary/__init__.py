from collections import defaultdict, Counter, namedtuple
from nltk.tree import Tree
from utils import do_nothing
from utils.types import NIL, F_RANDOM, F_RAND_CON, F_LEFT, F_RIGHT
from utils.math_ops import s_index, t_index
from data import SUB, SUP, brackets2RB
from data.continuous import bottom_paths, syn_tree

X_RGT = 1 << 0
X_DIR = 1 << 1
X_NEW = 1 << 2
X_ALL = X_RGT | X_RGT | X_NEW


def path_loc_gen(tree, path = ()):
    if syn_tree(tree) and (num := len(tree) - 1):
        for eid, child in enumerate(tree):
            child_path = path + (eid,)
            yield child_path, eid / num
            yield from path_loc_gen(child, child_path)

def tree_loc_path(lexical, tree, length):
    return tree, dict(path_loc_gen(tree)), bottom_paths(lexical, tree, length), length

def ori_right(factor, loc, path):
    if loc == 0:
        return True
    elif loc == 1:
        return False
    if isinstance(factor, dict):
        factor = factor[path]
    return loc < factor

from random import random
def extend_factor(factor, loc):
    if isinstance(factor, str):
        if factor == F_LEFT:
            factor = 0
        elif factor == F_RIGHT:
            factor = 1
        if factor == F_RAND_CON:
            factor = random()
        elif factor == F_RANDOM:
            factor = {p: random() for p in loc}
            factor[()] = random() # ROOT
    return factor

def signals(factor, tree, loc, lex_paths, length, every_n, l2i = do_nothing, xtype = X_ALL):
    factor    = extend_factor(factor, loc)
    primary   = Cell.layer(length)
    secondary = Cell.layer(length - 1)
    # build the triangular in a bottom-up fashion
    for cell, (path, lex) in zip(primary, lex_paths):
        cell.settle_in(factor, tree, loc, lex, path)
    layers_of_labels = []
    layers_of_xtypes = []
    while length > 1:
        for length in range(length, max(length - every_n, 1), -1):
            labels, xtypes = [], []
            for cell in primary[:length]:
                if status := cell.set_off():
                    dst, goods = status
                    secondary[dst].arrive(goods, factor, tree, loc)
                labels.append(cell.label)
                xtypes.append(cell.xtype & xtype)
                cell.reset()
            layers_of_labels.append(labels)
            layers_of_xtypes.append(xtypes)
            primary, secondary = secondary, primary

        if length == 2: break
        count = 0 # condense
        for cell in primary[:length - 1]:
            if status := cell.set_off():
                secondary[count].arrive(status[1])
                count += 1
            cell.reset()
        length = count
        primary, secondary = secondary, primary

    top = primary[0]
    layers_of_labels.append([top.label])
    layers_of_xtypes.append([top.xtype & xtype])
        
    return layers_of_labels, layers_of_xtypes


Goods = namedtuple('Goods', 'path, label, right, lhs, rhs')

class Cell:
    @classmethod
    def layer(cls, n):
        return tuple(Cell(i) for i in range(n))

    def __init__(self, sid):
        self._sid = sid
        self.reset()

    def reset(self):
        self._goods = None
        self._label = NIL
        self._xtype = 0

    def settle_in(self, factor, tree, loc, lexical, path):
        label = tree[path].label()
        if lexical:
            label = SUP + label
        else:
            self._xtype = X_NEW
        if path:
            loc = loc[path]
            right = ori_right(factor, loc, path)
            self._xtype |= X_DIR | X_RGT * right
            self._goods = Goods(path, label, right, loc, loc)
        self._label = label
                
    def set_off(self):
        if self._goods:
            return self._sid - (not self._goods.right), self._goods

    def arrive(self, goods, *params):
        if self._goods is None:
            # pass the necessary goods
            self._goods = goods
            self._label = goods.label
            self._xtype = goods.right * X_RGT | X_DIR
        else:
            factor, tree, loc = params
            l_path, l_label, _, lhs, _ = self._goods
            r_path, r_label, _, _, rhs = goods
            l_bound = lhs == 0
            r_bound = rhs == 1
            p_path = l_path[:-1]
            assert p_path == r_path[:-1]
            if l_bound and r_bound:
                self._xtype = X_NEW
                self._label = label = tree[p_path].label()
                if p_path:
                    loc = loc[p_path]
                    right = ori_right(factor, loc, p_path)
                    self._xtype = X_NEW | X_RGT * right
                    self._goods = Goods(p_path, label, right, loc, loc)
            else:
                if l_label[0] == SUB:
                    label = l_label
                elif r_label[0] == SUB:
                    label = r_label
                else:
                    label = tree[p_path].label()
                    if label[0] != SUB:
                        label = SUB + label
                if l_bound:
                    right = True
                elif r_bound:
                    right = False
                else:
                    if isinstance(factor, dict):
                        factor = factor[p_path]
                    right = ((lhs + rhs) / 2) < factor
                self._label = label
                self._xtype = X_RGT * right
                self._goods = Goods(l_path, label, right, lhs, rhs)

    @property
    def label(self):
        return self._label

    @property
    def xtype(self):
        return self._xtype



def explain_warnings(warnings, label_layers, tag_layer):
    templates = ['pos %(l)s and SUP %(p)s are not consistent',
                 'leftmost %(l)s directs away from %(p)s',    'rightmost %(r)s directs away from %(p)s', # bad
                 'left %(l)s goes through %(p)s',             'right %(r)s goes through %(p)s', # hard
                 'right %(r)s changes to %(p)s during relay', 'left %(l)s changes to %(p)s during relay', # okay strange 
                 'discard %(p)s',                             'root _%(p)s was a subphrase', # okay almost
                 '%(l)s and %(r)s join into %(p)s',           'lose subtree'] # err
    info = []
    for l, i, t in warnings:
        if i < 0:
            info.append(templates[i])
            continue
        if l == -1:
            data = dict(
                p = label_layers[0][i],
                l = tag_layer[i],
            )
        else:
            data = dict(
                p = label_layers[l][i],
                l = label_layers[l-1][i]   if l else tag_layer[i],
                r = label_layers[l-1][i+1] if l else tag_layer[i+1],
            )
        info.append(f'[{l}.{i}]', templates[t] % data)
    return info

templates = ['pos and SUP not consistent',
             'left/rightmost child directs away', # 1,2 not okay
             'child goes through <nil>s', # 3,4
             'tag changes during relay', # 5,6
             'discard non-<nil> parent',
             'root is a subphrase', # okay almost
             'children join into <nil>',
             'lose subtree']

def explain_one_error(err):
    return templates[err[2]] + f' at layer {err[0]}'

def sumup_warnings(warnings):
    cnt = defaultdict(int)
    for wtype, wcnt in Counter(warnings[:, 2]).items():
        if wtype <= 0: # without left or right attribute
            i = wtype
        elif wtype < 7: # left or right
            i = (wtype - 1) // 2 + 1
        else: # without again
            i = wtype - 3
        cnt[i] += wcnt
    for wtype, wcnt in cnt.items():
        yield templates[wtype], wcnt

import numpy as np
def warning_level(warnings):
    if isinstance(warnings, np.ndarray):
        warnings = warnings[:, 2]
    if len(warnings) == 0:
        return 0
    if 1 in warnings or 2 in warnings:
        return -2
    if -1 in warnings or -2 in warnings:
        return -1
    if 3 in warnings or 4 in warnings: # go through <nil>
        return 3
    if 1 in warnings or 2 in warnings or 7 in warnings: # go into paddings / discard something
        return 2
    return 1 # 0,5,6,8: pos/SUP, tag change, top is subtree

def get_tree_from_signals(word_layer, tag_layer, label_layers, right_layers, word_fn = brackets2RB):
    def _phrase(t): # -> list
        return t[:] if t.label()[0] == SUB else [t]

    warnings   = []
    last_layer = []
    if tag_layer is None:
        # SUP = ''
        for i, (w, s) in enumerate(zip(word_layer, label_layers[0])):
            last_layer.append(Tree(s, [word_fn(w)]))
    else:
        for i, (w, p, s) in enumerate(zip(word_layer, tag_layer, label_layers[0])):
            tagged_leaf = Tree(p, [word_fn(w)])
            if s[0] == SUB:
                tree = tagged_leaf
            elif s[0] == SUP:
                s = s[1:]
                if s == p:
                    tree = tagged_leaf
                else:
                    tree = Tree(s, [tagged_leaf])
                    if s != '_SUB':
                        warnings.append((-1, i, 0))
            else:
                tree = Tree(s, [tagged_leaf])
            last_layer.append(tree)

    leftmost, rightmost = 0, len(last_layer) - 1
    for layer_cnt, (right, upper) in enumerate(zip(right_layers, label_layers[1:])):
        this_layer = []
        rightmost -= 1
        smooth = len(right) == len(upper) + 1
        skipped_none = 0

        for i, p in enumerate(upper):
            if p[0] == SUP:
                p = p[1:]
            if smooth:
                l_child, r_child = last_layer[i], last_layer[i+1]
                lcrward, rclward =      right[i],  not right[i+1]
                left_relay       = l_child and lcrward
                right_relay      = r_child and rclward
            else:
                while True: # 2 or 3 hours ???
                    if i+skipped_none+1 == len(last_layer):
                        raise ValueError((layer_cnt, i+skipped_none, -1), last_layer, warnings)
                    l_child, r_child = last_layer[i+skipped_none], last_layer[i+skipped_none+1]
                    lcrward, rclward =      right[i+skipped_none],  not right[i+skipped_none+1]
                    left_relay       = l_child and lcrward
                    right_relay      = r_child and rclward
                    if left_relay or right_relay:
                        break
                    skipped_none += 1

            if i == leftmost and not lcrward: # left most shall be restrictly not nil and rightwards
                raise ValueError((layer_cnt, i, 1), last_layer, warnings)
                # warnings.append((layer_cnt, i, 1))
                # if r_child is None:
                #     this_layer.append(l_child)
                # else:
                #     this_layer.append(Tree(l_child.label(), ([l_child] if l_child.height() == 2 else l_child[:]) + _phrase(r_child)))
            elif i == rightmost and not rclward: # counterpart
                raise ValueError((layer_cnt, i, 2), last_layer, warnings)
                # warnings.append((layer_cnt, i, 2))
                # if l_child is None:
                #     this_layer.append(r_child)
                # else:
                #     this_layer.append(Tree(r_child.label(), _phrase(l_child) + ([r_child] if r_child.height() == 2 else r_child[:])))
            elif p == NIL: # phrase boundary -> nil
                if layer_cnt and left_relay and right_relay:
                    raise ValueError((layer_cnt, i, -2), last_layer, warnings)
                elif left_relay:
                    warnings.append((layer_cnt, i, 3))
                    this_layer.append(l_child)
                elif right_relay:
                    warnings.append((layer_cnt, i, 4))
                    this_layer.append(r_child)
                else:
                    this_layer.append(None)
            elif left_relay and right_relay: # word/phrase joint
                this_layer.append(Tree(p, _phrase(l_child) + _phrase(r_child) ))
            elif right_relay:
                if p[0] != SUB and not p.startswith(r_child.label()) and r_child.height() > 2: # should we keep the label?
                    # if r_child.label().startswith('_') or p.startswith('_'):
                    # if : # maybe not, less warning and more accurate
                    #     print('protect', r_child, p)
                    #     r_child = Tree(p, r_child)
                    # else:
                    r_child.set_label(p)
                    warnings.append((layer_cnt, i, 5))
                this_layer.append(r_child)
            elif left_relay:
                if p[0] != SUB and not p.startswith(l_child.label()) and l_child.height() > 2:
                    # if : # maybe not, less warning and more accurate
                    #     print('protect', l_child, p)
                    #     l_child = Tree(p, l_child)
                    # else:
                    l_child.set_label(p)
                    warnings.append((layer_cnt, i, 6))
                this_layer.append(l_child)
            else:
                warnings.append((layer_cnt, i, 7))
                this_layer.append(None)
        if len(word_layer) != sum(len(t.leaves()) for t in this_layer if t):
            raise ValueError((layer_cnt, -1, -1), last_layer, warnings)
        last_layer = this_layer
    root = last_layer[0]
    root_label = root.label()
    if root_label[0] == SUB:
        warnings.append((layer_cnt, 0, 8))
    return root, warnings

def after_to_tree(token_layer, tag_layer, label_layers, right_layers,
                    return_warnings = False,
                    on_warning      = None,
                    on_error        = None,
                    error_prefix    = '',
                    error_root      = 'S'):
    try:
        tree, warnings = get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers)
    except ValueError as e:
        error, last_layer, warnings = e.args
        if callable(on_error):
            on_error(error_prefix, explain_one_error(error))
        tree = Tree(error_root, [x for x in last_layer if x]) # Trust the model: TODO report failure rate
        warnings.append(error)
    if warnings and callable(on_warning) and tag_layer is not None:
        on_warning(explain_warnings(warnings, label_layers, tag_layer))
    if return_warnings: # [:, 2] > 8 is error
        warnings = np.asarray(warnings, dtype = np.int8)
        warnings.shape = (-1, 3)
        return tree, warnings
    return tree

def write_tensors(labels, xtypes, tensor_labels, tensor_xtypes, offset, paddings = None, vocab = None, skip_top = 0):
    tensor_vlen = tensor_labels.shape[0] + skip_top
    tensor_height, oset = t_index(tensor_vlen)
    assert oset == 0
    # assert tensor_labels.shape == tensor_xtypes.shape
    py_len = len(labels)
    py_height, oset = t_index(py_len)
    assert oset == 0
    assert py_len == len(xtypes)
    height_diff = tensor_height - py_height
    assert height_diff >= 0
    if paddings:
        l_bos, l_eos, x_bos, x_eos = paddings
        eos_d = height_diff - offset

    for src, (lbl, xty) in enumerate(zip(labels, xtypes)):
        if xty:
            lid, oset = t_index(src)
            dst = s_index(lid + height_diff, oset + offset) - skip_top
            if vocab is not None:
                lbl = vocab[lbl]
            tensor_labels[dst] = lbl
            tensor_xtypes[dst] = xty
            if paddings:
                if oset == 0:
                    start = dst - offset
                    tensor_labels[start:dst] = l_bos
                    tensor_xtypes[start:dst] = x_bos
                if oset == lid:
                    start = dst + 1
                    end = start + eos_d
                    tensor_labels[start:end] = l_eos
                    tensor_xtypes[start:end] = x_eos