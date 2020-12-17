from data.delta import preproc_cnf, Tree, defaultdict
from data.cross import _read_dpenn, draw_str_lines as _draw_str_lines

def clear_label(label, umark = '+', fmark = '@'):
    '''Most unaries are introduce by preproc_cnf/remove trace'''
    segs = []
    for seg in label.split(umark):
        if fmark in seg:
            seg = seg[:seg.index(fmark)]
        if seg not in segs:
            segs.append(seg)
    return umark.join(segs)

keep_str = lambda x: x
def signals(tree, w2i = keep_str, t2i = keep_str, l2i = keep_str):
    paths = []
    bottom = []
    bottom_label = []
    words, tags = [], []
    for wid, (word, tag) in enumerate(tree.pos()):
        path = tree.leaf_treeposition(wid)[:-2]
        paths.append(path)
        words.append(w2i(word))
        tags .append(t2i(tag ))
        bottom_label.append(l2i(clear_label(tree[path].label())))

    layers_of_labels = [bottom_label]
    layers_of_splits = []
    while len(paths) > 1:
        label_layer = []
        split_layer = []
        next_paths  = []
        child_cnt   = 0
        for pid, path in enumerate(paths):
            path_len = len(path)
            is_a_child = path[-1] > 0 and path_len == last_path_len
            complete = is_a_child and len(tree[path[:-1]]) == child_cnt + 1
            if complete:
                if child_cnt > 1:
                    split_layer = split_layer[:1-child_cnt]
                while child_cnt:
                    # print(pid, 'pop cousin:k', label_layer.pop())
                    label_layer.pop()
                    cousin_path = next_paths.pop()
                    assert cousin_path[:-1] == path[:-1]
                    child_cnt -= 1
                parent_path = path[:-1]
                label = tree[parent_path].label()
                next_paths.append(parent_path)
            else:
                next_paths.append(path)
                label = tree[path].label()
                if is_a_child:
                    child_cnt += 1
                else:
                    child_cnt = 1
                split_layer.append(pid)
            label_layer.append(l2i(clear_label(label)))
            last_path_len = path_len
        paths = next_paths
        assert len(label_layer) == len(split_layer)
        split_layer.append(pid + 1)
        # print(' '.join(label_layer))
        # print(split_layer)
        # print(' '.join(f'{label}({end-start})' for label, start, end in zip(label_layer, split_layer, split_layer[1:])))
        # print()
        layers_of_labels.append(label_layer)
        layers_of_splits.append(split_layer)
    return words, tags, layers_of_labels, layers_of_splits

def flatten_children(nodes):
    children = []
    for x in nodes:
        if isinstance(x, Tree):
            children.append(x)
        else:
            children.extend(x)
    return children

def flatten_children_with_weights(bottom, start, weights):
    children = []
    for nid, sub_tree in enumerate(bottom):
        mean, stdev = weights[start + nid] * 100
        sub_tree = [sub_tree] if isinstance(sub_tree, Tree) else flatten_children(sub_tree)
        children.append(Tree(f'{mean:.0f}%', sub_tree))
    return children

def get_tree_from_signals(word, tag, layers_of_labels, layers_of_splits, fall_back_root = None, layers_of_weights = None):
    bottom = []
    for w, t, label in zip(word, tag, layers_of_labels[0]):
        if w == '(':
            w = '-LRB-'
        elif w == ')':
            w = '-RRB-'
        if t in ('LRB', 'RRB'):
            t = '-' + t + '-'
        leave = Tree(t, [w])
        if label[0] != '#':
            leave = Tree(label, [leave])
        bottom.append(leave)

    # quit_on_error = False
    # bottom_len = len(bottom)

    for lid, (split_layer, label_layer) in enumerate(zip(layers_of_splits, layers_of_labels[1:])):
        new_bottom = []
        # leave_cnt = 0
        for label, start, end in zip(label_layer, split_layer, split_layer[1:]):

            if end - start == 1:
                if label[0] == '#' or bottom[start].label() == label:
                    sub_tree = bottom[start]
                else:
                    sub_tree = Tree(label, flatten_children(bottom[start:end]))
                # leave_cnt += len(sub_tree.leaves())
            elif label[0] == '#':
                assert fall_back_root is not None
                sub_tree = flatten_children(bottom[start:end])
                # leave_cnt += sum(len(x.leaves()) for x in sub_tree)
            else:
                if layers_of_weights is not None and lid < len(layers_of_weights):
                    children = flatten_children_with_weights(bottom[start:end], start, layers_of_weights[lid])
                else:
                    children = flatten_children(bottom[start:end])
                sub_tree = Tree(label, children)
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
            bottom = flatten_children_with_weights(bottom, 0, layers_of_weights[lid + 1])
        else:
            bottom = flatten_children(bottom)
        tree = Tree(fall_back_root, bottom)
    else:
        bottom = flatten_children(bottom)
        tree = bottom.pop()
        
    tree.un_chomsky_normal_form()
    if fall_back_root is None:
        return tree
    return tree, not bottom # and not quit_on_error

def draw_str_lines(tree):
    bottom_info, top_down, _ = _read_dpenn(tree)
    return _draw_str_lines(bottom_info, top_down)

class MAryX:
    def __init__(self, tree):
        preproc_cnf(tree)
        tree.collapse_unary(collapseRoot = True)
        self._raw_tree = tree

    @property
    def words(self):
        return self._raw_tree.leaves()

    @property
    def vocabs(self):
        words, tags, labels = (defaultdict(int) for i in range(3))
        for w, t in self._raw_tree.pos():
            words[w] += 1
            tags [t] += 1
        for tree in self._raw_tree.subtrees():
            if tree.height() > 2:
                labels[tree.label()] += 1
        return words, tags, labels

    def signals(self, *vocabs):
        return signals(self._raw_tree, *vocabs)

    # def __len__(self):
    #     return len(self._signals[0])