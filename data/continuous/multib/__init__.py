from data import SUP, brackets2RB
from data.mp import DM
from nltk.tree import Tree
from collections import defaultdict
from data.continuous import bottom_paths
from utils import do_nothing
from utils.str_ops import height_ratio

class MaryDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, segments, token, tag, label, fence, seg_length):
        for tokens, tags, labels, fences, seg_lengths in zip(token, tag, label, fence, seg_length):
            layers_of_label = []
            layers_of_fence = []
            label_start = 0
            fence_start = 0
            for l_cnt, (l_size, l_len) in enumerate(zip(segments, seg_lengths)):
                label_layer = tuple(i2l(i) for i in labels[label_start: label_start + l_len])
                layers_of_label.append(label_layer)
                if l_cnt:
                    layers_of_fence.append(fences[fence_start: fence_start + l_len + 1])
                    fence_start += l_size + 1
                else:
                    ln = l_len
                if l_len == 1:
                    break
                label_start += l_size
            wd = [i2w[i] for i in tokens[:ln]]
            tg = [i2t[i] for i in   tags[:ln]]
            tree, _ = get_tree_from_signals(wd, tg, layers_of_label, layers_of_fence, 'VROOT')
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        start = seg_id * seg_size
        if start < batch_size:
            return args[:1] + tuple(x[start: (seg_id + 1) * seg_size] for x in args[1:])

tree_path = lambda lexical, tree, length: (tree, bottom_paths(lexical, tree, length))

def signals(tree, lex_paths, l2i = do_nothing):
    paths, bottom_label = [], []
    for eid, (path, lexical) in enumerate(lex_paths):
        label = tree[path].label()
        if lexical:
            label = SUP + label
        bottom_label.append(l2i(label))
        paths.append(path)
        
    def append_to_layer(eid, label, path):
        split_layer.append(eid)
        label_layer.append(label)
        npath_layer.append(path)
        
    END = (-1,)
    layers_of_labels = [bottom_label]
    layers_of_splits = []
    while len(paths) > 1:
        label_layer = []
        split_layer = [0]
        npath_layer = []
        child_count = 1
        for eid, (l_path, r_path) in enumerate(zip(paths, paths[1:] + [END]), 1):
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
        assert len(npath_layer) == len(label_layer) == len(split_layer) - 1
        layers_of_labels.append(label_layer)
        layers_of_splits.append(split_layer)
        bottom_label = label_layer
        paths = npath_layer
    return layers_of_labels, layers_of_splits


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

def flatten_children(nodes):
    children = []
    for x in nodes:
        if isinstance(x, Tree):
            children.append(x)
        else:
            children.extend(x)
    return children

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
                          fall_back_root           = None,
                          layers_of_weights        = None,
                          layers_of_fence_vote     = None,
                          mark_np_without_dt_child = False,
                          word_fn                  = brackets2RB):
    bottom = []
    add_weight_base = layers_of_weights is not None
    balancing_bottom_sub = add_weight_base and any(x[0] not in '#_' for x in layers_of_labels[0])
    add_fence_vote_base = layers_of_fence_vote is not None
    unary_chars = '│' + func_unary_char 
    for w, t, label in zip(word, tag, layers_of_labels[0]):
        leaf = Tree(t, [word_fn(w)])
        if label[0] not in '#_':
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
        balancing_sub = add_weight_base and any(x[0] not in '#_' for x in label_layer)
        add_fence_vote = add_fence_vote_base and lid < len(layers_of_fence_vote)
        if add_fence_vote:
            bottom = flatten_layer_with_fence_vote(bottom, layers_of_fence_vote[lid])

        for label, start, end in zip(label_layer, split_layer, split_layer[1:]):
            if end - start == 1: # unary
                sub_tree = bottom[start]
                if label[0] in '#_' or (unary_label_match(sub_tree[0], label) if add_fence_vote else (sub_tree.label() == label)):
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
                assert fall_back_root is not None
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
                # leave_cnt += len(sub_tree.leaves())
            # print(str(sub_tree))
            new_bottom.append(sub_tree)
        # import pdb; pdb.set_trace()
        bottom = new_bottom

        # if leave_cnt != bottom_len:
        #     quit_on_error = True
        # if quit_on_error:
        #     break
        
    if fall_back_root is None:
        tree = bottom.pop()
        assert not bottom
    elif len(bottom) > 1:
        if layers_of_weights and layers_of_weights[lid + 1]: # never be here
            bottom, _ = flatten_children_with_weights(bottom, 0, layers_of_weights[lid + 1])
        else:
            bottom = flatten_children(bottom)
        tree = Tree(fall_back_root, bottom)
    else:
        bottom = flatten_children(bottom)
        tree = bottom.pop()
    
    if add_weight_base or add_fence_vote_base:
        tree.un_chomsky_normal_form(unaryChar = func_unary_char)
    if fall_back_root is None:
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
