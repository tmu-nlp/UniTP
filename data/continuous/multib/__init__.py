from nltk.tree import Tree
from collections import defaultdict
from data import SUP, SUBS, brackets2RB
from data.continuous import bottom_paths, flatten_children
from utils.str_ops import height_ratio
from utils import do_nothing

tree_path = lambda lexical, tree, length: (tree, bottom_paths(lexical, tree, length))

from random import random
def append_sub(subword, msub, max_height, path, bpaths, bottom):
    lhs, rhs = random_subword_path(subword, msub, max_height, path + (len(bottom),))
    if isinstance(rhs, tuple): # leave
        bpaths.append(rhs)
    else:
        bpaths.extend(rhs)
    bottom.append(lhs)

def char_leaf(chars, path):
    if len(chars) == 1:
        return Tree(SUP, [chars]), path
    leaves = []
    bpaths = []
    for eid, x in enumerate(chars):
        leaves.append(Tree(SUP, [x]))
        bpaths.append(path + (eid,))
    return Tree(SUP, leaves), bpaths

def random_subword_path(word, msub = 0, max_height = None, path = ()):
    bottom = []
    bpaths = []
    if max_height and max_height == len(path) + (len(word) > 1):
        return char_leaf(word, path)

    subword = word[0]
    for char in word[1:]:
        if msub and random() < msub:
            append_sub(subword, msub, max_height, path, bpaths, bottom)
            subword = char
        else:
            subword += char

    if not bottom:
        return char_leaf(word, path)

    append_sub(subword, msub, max_height, path, bpaths, bottom)
    return Tree(SUP, bottom), bpaths

def char_signals(words, msub = 0, max_subword_height = None):
    if max_subword_height:
        assert max_subword_height > 1
    bottom = []
    wpaths = []
    for eid, word in enumerate(words):
        tree, path = random_subword_path(word, msub, max_subword_height, (eid,))
        bottom.append(tree)
        if isinstance(path, list):
            wpaths.extend(path)
        else:
            wpaths.append(path)
    return Tree(SUP, bottom), tuple((p, False) for p in wpaths)

def get_words_from_tree(tree):
    return [''.join(wd.leaves()) for wd in tree]


_END = [(-1,)]
def j_fn(bottom_label, paths, tree, l2i, offset = 1): # advantage@fewer_signals
    label_layer = []
    joint_layer = []
    npath_layer = []
    joint_count = 1
    for eid, (l_path, r_path) in enumerate(zip(paths, paths[1:] + _END)):
        if (p_path := l_path[:-1]) == r_path[:-1] and l_path[-1] + 1 == r_path[-1]:
            joint_count += 1
            joint_layer.append(eid + offset)
            if joint_count == len(tree[p_path]):
                label_layer.append(l2i(tree[p_path].label()))
                npath_layer.append(p_path)
                joint_count = 0 # impossible to rejoin rightwards
        elif joint_count > 1:
            joint_layer = joint_layer[:1 - joint_count]
            label_layer.extend(bottom_label[eid + 1 - joint_count:eid + 1])
            npath_layer.extend(paths       [eid + 1 - joint_count:eid + 1])
            joint_count = 1
        elif joint_count == 1:
            label_layer.append(bottom_label[eid])
            npath_layer.append(l_path)
        else:
            joint_count += 1
    return label_layer, joint_layer, npath_layer

def s_fn(bottom_label, paths, tree, l2i): # advantage@decoding
    def append_to_layer(eid, label, path):
        split_layer.append(eid)
        label_layer.append(label)
        npath_layer.append(path)
    label_layer = []
    split_layer = [0]
    npath_layer = []
    child_count = 1
    for eid, (l_path, r_path) in enumerate(zip(paths, paths[1:] + _END), 1):
        if (p_path := l_path[:-1]) == r_path[:-1] and l_path[-1] + 1 == r_path[-1]:
            child_count += 1
        else:
            if child_count == len(tree[p_path]):
                path  = p_path
                label = l2i(tree[path].label())
            else: # lack direct children
                for i in range(eid - child_count, eid - 1):
                    append_to_layer(i + 1, bottom_label[i], paths[i])
                path  = l_path
                label = bottom_label[eid - 1]
            append_to_layer(eid, label, path)
            child_count = 1
    return label_layer, split_layer, npath_layer
    
def signals(tree, lex_paths, l2i = do_nothing, joint = None):
    paths, bottom_label = [], []
    for path, lexical in lex_paths:
        label = tree[path].label()
        if lexical:
            label = SUP + label
        bottom_label.append(l2i(label))
        paths.append(path)
        
    layers_of_labels = [bottom_label]
    layers_of_struct = []
    while len(paths) > 1:
        if joint is None: # self validation
            sl, ss, sp = s_fn(bottom_label, paths, tree, l2i)
            jl, jj, jp = j_fn(bottom_label, paths, tree, l2i)
            assert len(sp) == len(sl) == len(ss) - 1
            assert len(jp) == len(jl)
            assert sl == jl and sp == jp and not (set(ss) & set(jj))
            if (set(ss) | set(jj)) != set(range(max(ss) + 1)):
                print(ss)
                print(jj)
                breakpoint()
            bottom_label = sl
            paths = sp
            continue
        (label_layer, struct_layer,
         npath_layer) = (s_fn, j_fn)[joint](bottom_label, paths, tree, l2i)
        layers_of_labels.append(label_layer)
        layers_of_struct.append(struct_layer)
        bottom_label = label_layer
        paths = npath_layer
    return layers_of_labels, layers_of_struct


def coord_vote(units, fence_location):
    if units.sum() > 0:
        pro_num_ratio = (units > 0).sum() / units.size
        unit_location = units.argmax()
    else:
        pro_num_ratio = (units <= 0).sum() / units.size
        unit_location = units.argmin()
    if fence_location > unit_location:
        return f'«{fence_location - unit_location - 1}{height_ratio(pro_num_ratio)}'
    return f'{height_ratio(pro_num_ratio)}{unit_location - fence_location}»'

func_unary_char = '&'
def flatten_children_with_weights(bottom, start, weights, bar = '│'):
    children = []
    head_child = None
    max_weight = 0
    for nid, sub_tree in enumerate(bottom):
        mean, _ = weights[start + nid]
        label = f'{mean * 100:.0f}%'
        if isinstance(sub_tree, Tree):
            unary_label = sub_tree.label()
            if bar in unary_label:
                sub_tree.set_label(label + func_unary_char + unary_label)
            else:
                sub_tree = Tree(label, [sub_tree])
        else:
            sub_tree = Tree(label, flatten_children(sub_tree))
        children.append(sub_tree)
        if mean > max_weight:
            max_weight = mean
            head_child = sub_tree
    return children, head_child

def flatten_layer_with_fence_vote(bottom, fence_vote, bar = '│'):
    children = []
    for nid, sub_tree in enumerate(bottom):
        if nid == 0:
            lhs, rhs = fence_vote[:2]
            lhs = coord_vote(lhs, nid)
            rhs = coord_vote(rhs, nid + 1)
            lhs_len = len(lhs)
            rhs_len = len(rhs)
            if lhs_len > rhs_len:
                label = lhs + bar + rhs + ' ' * (lhs_len - rhs_len)
            elif lhs_len < rhs_len:
                label = lhs + ' ' * (rhs_len - lhs_len) + bar + rhs
            else:
                label = lhs + bar + rhs 
        else:
            rhs = fence_vote[nid + 1]
            rhs = coord_vote(rhs, nid + 1)
            lhs = ' ' * len(rhs)
            label = lhs + bar + rhs
        if isinstance(sub_tree, Tree):
            unary_label = sub_tree.label()
            if bar in unary_label:
                sub_tree.set_label(' ' + label + func_unary_char + unary_label) # L%FL%FLLL
            else:
                sub_tree = Tree(' ' + label, [sub_tree])
        else:
            sub_tree = Tree(' ' + label, flatten_children(sub_tree))
        children.append(sub_tree)
    return children

def unary_label_match(tree, label):
    while tree.height() > 2:
        if tree.label() == label:
            return True
        if len(tree) == 1:
            tree = tree[0]
        else:
            return False
    return False

def get_tree_from_signals(word, tag, layers_of_labels, layers_of_splits,
                          layers_of_weights        = None,
                          layers_of_fence_vote     = None,
                          fallback_label           = None,
                          mark_np_without_dt_child = False,
                          word_fn                  = brackets2RB):
    bottom = []
    add_weight_base = layers_of_weights is not None
    add_fence_vote_base = layers_of_fence_vote is not None
    balancing_bottom_sub = add_weight_base and any(x[0] not in SUBS for x in layers_of_labels[0])
    unary_chars = '│' + func_unary_char 
    for w, t, label in zip(word, tag, layers_of_labels[0]):
        leaf = Tree(t, [word_fn(w)])
        if label[0] not in SUBS:
            leaf = Tree(label, [leaf])
        elif balancing_bottom_sub:
            leaf = Tree('│', [leaf])
        bottom.append(leaf)

    if add_weight_base:
        headedness_stat = {}
    else:
        headedness_stat = None

    for lid, (split_layer, label_layer) in enumerate(zip(layers_of_splits, layers_of_labels[1:])):
        new_bottom = []
        # leave_cnt = 0
        add_weight = add_weight_base and lid < len(layers_of_weights)
        balancing_sub = add_weight_base and any(x[0] not in SUBS for x in label_layer)
        add_fence_vote = add_fence_vote_base and lid < len(layers_of_fence_vote)
        if add_fence_vote:
            bottom = flatten_layer_with_fence_vote(bottom, layers_of_fence_vote[lid])

        for label, start, end in zip(label_layer, split_layer, split_layer[1:]):
            if end - start == 1: # unary
                sub_tree = bottom[start]
                if label[0] in SUBS or (unary_label_match(sub_tree[0], label) if add_fence_vote else (sub_tree.label() == label)):
                    # relay sub
                    if isinstance(sub_tree, Tree) and balancing_sub:
                        relay_label = sub_tree.label()
                        if '│' in relay_label:
                            sub_tree.set_label(unary_chars + unary_chars + relay_label)
                        else:
                            sub_tree = Tree(unary_chars + '│', [sub_tree])
                else:
                    sub_tree = Tree(label, flatten_children(bottom[start:end]))
                # leave_cnt += len(sub_tree.leaves())
            elif label[0] == '#':
                assert fallback_label is not None
                if balancing_sub:
                    children = flatten_children(bottom[start:end])
                    if add_weight:
                        children, head_child = flatten_children_with_weights(children, start, layers_of_weights[lid])
                        sub_tree = Tree(label, children)
                    else:
                        sub_tree = Tree(unary_chars + '│', sub_tree)
                else:
                    sub_tree = flatten_children(bottom[start:end])
                # leave_cnt += sum(len(x.leaves()) for x in sub_tree)
            else:
                children = bottom[start:end]
                if add_weight:
                    children, head_child = flatten_children_with_weights(children, start, layers_of_weights[lid])
                    sub_tree = Tree(label, children) # +2
                    try:
                        head_label = head_child[0].label()
                        if mark_np_without_dt_child and label == 'NP' and not any(x[0].label() == 'DT' for x in children):
                            head_label += '*'
                    except:
                        print(head_child[0])
                        import pdb; pdb.set_trace()
                    if '│' in head_label or func_unary_char in head_label:
                        import pdb; pdb.set_trace()
                    if label in headedness_stat:
                        label_cnt, head_cnts = headedness_stat[label]
                    else:
                        label_cnt = 0
                        head_cnts = defaultdict(int)
                    label_cnt += 1
                    head_cnts[head_label] += 1
                    headedness_stat[label] = label_cnt, head_cnts
                else:
                    children = flatten_children(children)
                    sub_tree = Tree(label, children) if label[0] != '_' else children
            new_bottom.append(sub_tree)
        bottom = new_bottom
        
    if fallback_label is None:
        tree = bottom.pop()
        assert not bottom
    elif len(bottom) == 1:
        bottom = flatten_children(bottom)
        tree = bottom.pop()
    else:
        bottom = flatten_children(bottom)
        tree = Tree(fallback_label, flatten_children(bottom))
    
    if add_weight_base or add_fence_vote_base:
        tree.un_chomsky_normal_form(unaryChar = func_unary_char)
    if fallback_label is None:
        return tree
    if add_weight_base:
        return tree, not bottom, headedness_stat
    return tree, not bottom # and not quit_on_error


#     def check_signals(self):
#         origin = clear_raw_tree(self._raw_tree)
#         tree     = get_tree_from_signals(*self.signals())
#         sub_tree = get_tree_from_signals(*self.sub_signals())
#         tree    .collapse_unary(collapseRoot = True)
#         sub_tree.collapse_unary(collapseRoot = True)
#         if not origin == tree == sub_tree:
#             origin = ('\n'.join(draw_str_lines(origin)))
#             tree = ('\n'.join(draw_str_lines(tree)))
#             sub_tree = ('\n'.join(draw_str_lines(sub_tree)))
#             if origin != tree:
#                 print(origin)
#                 print(tree)
#                 breakpoint()
#             if origin != sub_tree:
#                 print(origin)
#                 print(sub_tree)
#                 breakpoint()

# def clear_raw_tree(raw):
#     label = raw.label()
#     if raw.height() < 3:
#         if label in ('LRB', 'RRB'):
#             return Tree('-' + label + '-', raw[:])
#         return raw
#     if label[0] in '_#':
#         return [clear_raw_tree(x) for x in raw]
#     segs = []
#     for x in label.split('+'):
#         if (at := x.find('@')) >= 0:
#             x = x[:at]
#         if x not in segs:
#             segs.append(x)
#     return Tree('+'.join(segs), flatten_children(clear_raw_tree(sub) for sub in raw))
