from collections import namedtuple
from nltk.tree import Tree
from utils import do_nothing
from utils.types import F_RANDOM, F_CNF, F_LEFT, F_RIGHT
from data import NIL, SUB, SUP, SUBS, brackets2RB
from data.continuous import bottom_paths, syn_tree, flatten_children

X_RGT = 1 << 0
X_DIR = 1 << 1
X_NEW = 1 << 2
X_BND = 1 << 3


def path_loc_gen(tree, path = ()):
    if syn_tree(tree) and (num := len(tree) - 1):
        for eid, child in enumerate(tree):
            child_path = path + (eid,)
            yield child_path, eid / num
            yield from path_loc_gen(child, child_path)

def tree_loc_path_length(lexical, tree, length):
    return tree, dict(path_loc_gen(tree)), bottom_paths(lexical, tree, length), length

def ori_right(factor, loc, path):
    if loc == 0:
        return True, X_BND
    elif loc == 1:
        return False, X_BND
    if isinstance(factor, dict):
        factor = factor[path]
    return loc < factor, 0

from random import random, betavariate
def extend_factor(factor, loc):
    if isinstance(factor, str):
        if factor == F_LEFT:
            factor = 0
        elif factor == F_RIGHT:
            factor = 1
        elif factor == F_RANDOM: # F_CON
            factor = {p: random() for p in loc}
            factor[()] = random()
    elif isinstance(factor, tuple):
        lhs, rhs = factor
        if lhs == F_CNF:
            factor = {p: random() < rhs for p in loc}
        else:
            factor = {p: betavariate(lhs, rhs) for p in loc}
        factor[()] = random()
    return factor

def signals(factor, tree, loc, lex_paths, length, every_n, l2i = do_nothing, xtype = -1):
    if every_n < 1: every_n += length
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
                labels.append(l2i(cell.label))
                xtypes.append(cell.xtype & xtype)
                cell.reset()
            layers_of_labels.append(labels)
            layers_of_xtypes.append(xtypes)
            primary, secondary = secondary, primary

        if length == 2: break
        count = 0 # condense
        for cell in primary[:length - 1]:
            if status := cell.set_off():
                secondary[count].arrive(status[1], cell.xtype)
                count += 1
            cell.reset()
        length = count
        primary, secondary = secondary, primary

    top = primary[0]
    layers_of_labels.append([l2i(top.label)])
    layers_of_xtypes.append([top.xtype & xtype])
        
    return layers_of_labels, layers_of_xtypes

def get_tree_from_signals(word_layer, tag_layer, label_layers, right_layers,
                          fallback_label = None, word_fn = brackets2RB, keep_sub = False):
    warnings   = []
    primary, secondary = [], []
    for eid, (word, label) in enumerate(zip(word_layer, label_layers[0])):
        if tag_layer is None:
            leaf = Tree(label, [word_fn(word) if callable(word_fn) else word])
        else:
            tag  = tag_layer[eid]
            leaf = Tree(tag, [word_fn(word)])
            if label[0] == SUP: #TAG
                if label[1:] != tag:
                    warnings.append((-1, eid, 0))
            elif label[0] != SUB: # unary
                leaf = Tree(label, [leaf])
        primary.append(Node(eid).settle_in(leaf, keep_sub))
        secondary.append(Node(eid))

    for layer_cnt, (right, label) in enumerate(zip(right_layers, label_layers[1:])):
        this_len = len(right)
        next_len = len(label)
        if fallback_label and this_len == 1: break

        if redirect := this_len - 1 > next_len:
            last_r = False
            joints = 0

        for node, r in zip(primary, right):
            if tree := node.tree:
                if redirect:
                    if last_r and not r:
                        joints += 1
                    dst = node.sid - joints
                    last_r = r
                else:
                    dst = node.sid - (not r)
                if dst < 0 or dst == next_len:
                    return Tree(fallback_label, flatten_children(n.tree for n in primary[:this_len] if n.tree)), (layer_cnt, node.sid, warnings)
                secondary[dst].settle_in(tree, keep_sub, label[dst])
        for node in primary[:this_len]:
            node.reset()
        primary, secondary = secondary, primary

    tree = primary[0].tree
    if fallback_label:
        if not isinstance(tree, Tree):
            tree = Tree(fallback_label, tree)
            warnings.append((layer_cnt + 1, 0, 1))
        return tree, warnings
    assert not warnings
    return tree


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
            right, boundary = ori_right(factor, loc, path)
            self._xtype |= X_DIR | X_RGT * right | boundary
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
            if len(params) == 1:
                self._xtype, = params
            else:
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
                self._label = label = tree[p_path].label()
                if p_path:
                    loc = loc[p_path]
                    right, boundary = ori_right(factor, loc, p_path)
                    self._xtype = X_NEW | X_RGT * right | X_DIR | boundary
                    self._goods = Goods(p_path, label, right, loc, loc)
                else:
                    self._xtype = X_NEW | X_RGT
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
                self._xtype = X_RGT * right | X_DIR | X_BND * (l_bound or r_bound)
                self._goods = Goods(l_path, label, right, lhs, rhs)

    @property
    def label(self):
        return self._label

    @property
    def xtype(self):
        return self._xtype

as_children = lambda x: [x] if isinstance(x, Tree) else x # Tree is list

class Node:
    def __init__(self, sid):
        self._sid = sid
        self.reset()

    def reset(self):
        self._tree = None

    def settle_in(self, tree, keep_sub, label = None):
        if self._tree is None:
            self._tree = tree
        else:
            tree = as_children(self._tree) + as_children(tree)
            if keep_sub or label[0] not in SUBS:
                tree = Tree(label, tree)
            self._tree = tree
        return self

    @property
    def tree(self):
        return self._tree

    @property
    def sid(self):
        return self._sid


# def explain_warnings(warnings, label_layers, tag_layer):
#     templates = ['pos %(l)s and SUP %(p)s are not consistent',
#                  'leftmost %(l)s directs away from %(p)s',    'rightmost %(r)s directs away from %(p)s', # bad
#                  'left %(l)s goes through %(p)s',             'right %(r)s goes through %(p)s', # hard
#                  'right %(r)s changes to %(p)s during relay', 'left %(l)s changes to %(p)s during relay', # okay strange 
#                  'discard %(p)s',                             'root _%(p)s was a subphrase', # okay almost
#                  '%(l)s and %(r)s join into %(p)s',           'lose subtree'] # err
#     info = []
#     for l, i, t in warnings:
#         if i < 0:
#             info.append(templates[i])
#             continue
#         if l == -1:
#             data = dict(
#                 p = label_layers[0][i],
#                 l = tag_layer[i],
#             )
#         else:
#             data = dict(
#                 p = label_layers[l][i],
#                 l = label_layers[l-1][i]   if l else tag_layer[i],
#                 r = label_layers[l-1][i+1] if l else tag_layer[i+1],
#             )
#         info.append(f'[{l}.{i}]', templates[t] % data)
#     return info

# templates = ['pos and SUP not consistent',
#              'left/rightmost child directs away', # 1,2 not okay
#              'child goes through <nil>s', # 3,4
#              'tag changes during relay', # 5,6
#              'discard non-<nil> parent',
#              'root is a subphrase', # okay almost
#              'children join into <nil>',
#              'lose subtree']

# def explain_one_error(err):
#     return templates[err[2]] + f' at layer {err[0]}'

# def after_to_tree(token_layer, tag_layer, label_layers, right_layers,
#                     return_warnings = False,
#                     on_warning      = None,
#                     on_error        = None,
#                     error_prefix    = '',
#                     error_root      = 'S'):
#     try:
#         tree, warnings = get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers)
#     except ValueError as e:
#         error, tree_layer, warnings = e.args
#         if callable(on_error):
#             on_error(error_prefix, explain_one_error(error))
#         tree = Tree(error_root, [x for x in tree_layer if x]) # Trust the model: TODO report failure rate
#         warnings.append(error)
#     if warnings and callable(on_warning) and tag_layer is not None:
#         on_warning(explain_warnings(warnings, label_layers, tag_layer))
#     if return_warnings: # [:, 2] > 8 is error
#         warnings = np.asarray(warnings, dtype = np.int8)
#         warnings.shape = (-1, 3)
#         return tree, warnings
#     return tree

# def _phrase(t):
#     return t[:] if t.label()[0] == SUB else [t]
        # next_layer = []
        # nil_count = 0

        # if smooth := len(right) == len(upper) + 1:
        #     if not right[0]: # left most shall be restrictly not nil and rightwards
        #         return Tree(fallback_label, tree_layer), ((layer_cnt, 0, 1), warnings)
        #     elif right[-1]: # counterpart
        #         return Tree(fallback_label, tree_layer), ((layer_cnt, -1, 2), warnings)

    #     for i, label in enumerate(upper):
    #         if label[0] == SUP:
    #             label = label[1:]
    #         if smooth:
    #             l_child, r_child = tree_layer[i], tree_layer[i+1]
    #             lcrward, rclward =      right[i],  not right[i+1]
    #             left_relay       = l_child and lcrward
    #             right_relay      = r_child and rclward
    #         else:
    #             while True:
    #                 if i + nil_count + 1 == len(tree_layer):
    #                     return Tree(fallback_label, tree_layer), ((layer_cnt, -1, 2), warnings)
    #                 l_child, r_child = tree_layer[i+nil_count], tree_layer[i+nil_count+1]
    #                 lcrward, rclward =      right[i+nil_count],  not right[i+nil_count+1]
    #                 left_relay       = l_child and lcrward
    #                 right_relay      = r_child and rclward
    #                 if left_relay or right_relay:
    #                     break
    #                 nil_count += 1

    #         if label == NIL: # phrase boundary -> nil
    #             if layer_cnt and left_relay and right_relay:
    #                 return Tree(fallback_label, tree_layer), ((layer_cnt, i, -2), warnings)
    #             elif left_relay:
    #                 warnings.append((layer_cnt, i, 3))
    #                 next_layer.append(l_child)
    #             elif right_relay:
    #                 warnings.append((layer_cnt, i, 4))
    #                 next_layer.append(r_child)
    #             else:
    #                 next_layer.append(None)
    #         elif left_relay and right_relay: # word/phrase joint
    #             next_layer.append(Tree(label, _phrase(l_child) + _phrase(r_child) ))
    #         elif right_relay:
    #             if label[0] != SUB and not label.startswith(r_child.label()) and r_child.height() > 2:
    #                 r_child.set_label(label)
    #                 warnings.append((layer_cnt, i, 5))
    #             next_layer.append(r_child)
    #         elif left_relay:
    #             if label[0] != SUB and not label.startswith(l_child.label()) and l_child.height() > 2:
    #                 l_child.set_label(label)
    #                 warnings.append((layer_cnt, i, 6))
    #             next_layer.append(l_child)
    #         else:
    #             warnings.append((layer_cnt, i, 7))
    #             next_layer.append(None)
    #     if len(word_layer) != sum(len(t.leaves()) for t in next_layer if t):
    #         return Tree(fallback_label, tree_layer), ((layer_cnt, -1, -1), warnings)
    #     tree_layer = next_layer
    # root = tree_layer[0]
    # root_label = root.label()
    # if root_label[0] == SUB:
    #     warnings.append((layer_cnt, 0, 8))
    # return root, warnings