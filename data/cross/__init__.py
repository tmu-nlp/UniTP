from collections import namedtuple, defaultdict, Counter
from utils.param_ops import get_sole_key
TopDown = namedtuple('TopDown', 'label, children')
C_VROOT = 'VROOT'

def validate(bottom_info, top_down,
             single_attachment = True,
             check_redundancy  = True):
    existing_bids = set(bid for bid, _, _ in bottom_info)
    nids, cids = [0], set()
    bid_refs, nid_refs = set(), set()
    while nids:
        for nid in nids:
            if nid > 0:
                assert nid in existing_bids, f'Refering to non-existing bid({nid})'
                if single_attachment:
                    assert nid not in bid_refs, f'Repeating bid({nid})'
                bid_refs.add(nid)
            else:
                assert nid in top_down, f'Lacking nid({nid}) in top_down.keys'
                if single_attachment:
                    assert nid not in nid_refs, f'Multi-attachment [configurable] nid({nid})'
                nid_refs.add(nid)
                for cid in top_down[nid].children:
                    cids.add(cid)
        nids = cids
        cids = set()
    if check_redundancy:
        redundancy = existing_bids - bid_refs
        assert not redundancy, 'Redundant bid(s): ' + ', '.join(str(x) for x in redundancy)
        redundancy = top_down.keys() - nid_refs
        assert not redundancy, 'Redundant nid(s): ' + ', '.join(str(x) for x in redundancy)

from random import random
from utils.types import F_CON, F_DEP, F_RANDOM, F_LEFT
from copy import deepcopy
class Signal:
    @classmethod
    def set_binary(cls):
        from data.cross.binary import cross_signals
        def func(factor, more_sub, bottom, node2tag, bottom_unary, top_down, l2i, dep):
            if factor != F_DEP:
                if factor == F_RANDOM:
                    factor = random()
                dep = None
            top_down = deepcopy(top_down)
            if more_sub:
                top_down = add_subs(top_down, bottom_unary, more_sub)
            return cross_signals(bottom, node2tag, bottom_unary, top_down, factor, dep, l2i = l2i)
        cls.__binary = func

    @classmethod
    def set_multib(cls):
        from data.cross.multib import cross_signals
        def func(factor, more_sub, bottom, node2tag, bottom_unary, top_down, l2i, dep):
            if factor == F_DEP:
                assert isinstance(dep, dict)
            elif isinstance(factor, dict):
                dep = factor # check static for cache?
                factor = F_DEP
            if factor == F_DEP:
                # if self._dep_signals is None TODO 
                return cross_signals(bottom, node2tag, bottom_unary, top_down, factor, l2i, _new_dep(dep))
            if 0 < more_sub < 1 and factor != F_CON:
                top_down = add_subs(top_down, bottom_unary, more_sub)
            return cross_signals(bottom, node2tag, bottom_unary, top_down, factor, l2i)
        cls.__multib = func

    @classmethod
    def from_tiger_graph(cls, graph, *args, **kw_args):
        from data.cross.tiger import read_tree
        return cls(*read_tree(graph), *args, **kw_args)

    @classmethod
    def from_disco_penn(cls, tree, *args, **kw_args):
        from data.cross.dptb import read_tree
        # args = replace_args_kwargs(_dep_n_prefix, 1, args, 'dep', kw_args)
        return cls(*read_tree(tree), *args, **kw_args)

    def __init__(self, bottom, top_down, v2is = None, dep = None):
        self._tree = bottom, deepcopy(top_down)
        word, bottom_nodes, node2tag, bottom_unary = _pre_proc(bottom, top_down, dep = dep)
        self._word = word
        self._gaps = None
        if v2is is None:
            bottom_tag = [node2tag[t] for t in bottom_nodes]
            l2i = None
        else:
            w2i, t2i, l2i = v2is
            bottom_tag = [t2i(node2tag[t]) for t in bottom_nodes]
            word = [w2i(w) for w in word]
        if dep is not None:
            _pre_dep(dep, node2tag)

        self._word_tag = word, bottom_tag
        self._materials = bottom_nodes, top_down, bottom_unary, node2tag, l2i, dep
        self._balanced_top_down = None

    @property
    def has_signals(self):
        return any(self._materials[1:3])

    @property
    def gap(self):
        return max(self.gaps.values())

    @property
    def gaps(self):
        if self._gaps is None:
            bottom_nodes, top_down = self._materials[:2]
            self._gaps = gap_degree(bottom_nodes, top_down, None, True)
        return self._gaps

    @property
    def text(self):
        return self._word

    @property
    def word_tag(self):
        return self._word_tag
        
    def binary(self, factor = F_RANDOM, balancing_sub = False, more_sub = 0):
        bottom_nodes, top_down, bottom_unary, node2tag, l2i, dep = self._stratify(balancing_sub)
        return Signal.__binary(factor, more_sub, bottom_nodes, node2tag, bottom_unary, top_down, l2i, dep)

    def multib(self, factor = F_LEFT, balancing_sub = False, more_sub = 0):
        bottom_nodes, top_down, bottom_unary, node2tag, l2i, dep = self._stratify(balancing_sub)
        return Signal.__multib(factor, more_sub, bottom_nodes, node2tag, bottom_unary, top_down, l2i, dep)

    def _stratify(self, balancing_sub = False):
        bottom_nodes, top_down, bottom_unary, node2tag, l2i, dep = self._materials
        if balancing_sub:
            if self._balanced_top_down is None:
                self._balanced_top_down = add_efficient_subs(top_down, bottom_unary)
            top_down = self._balanced_top_down
        return bottom_nodes, top_down, bottom_unary, node2tag, l2i, dep

def height_gen(top_down, root_id):
    max_height = -1
    if root_id in top_down: # pre-terminals at 0
        for node in top_down[root_id].children:
            for cid, height in height_gen(top_down, node):
                yield cid, height
                if height > max_height:
                    max_height = height
        yield root_id, max_height + 1

def has_label(top_down, root_id, label):
    if root_id in top_down:
        if top_down[root_id].label == label:
            return True
        for node in top_down[root_id].children:
            if has_label(top_down, node, label):
                return True
    return False
    
def add_efficient_subs(top_down, bottom_unary, root_id = 0, sub_prefix = '_', sub_suffix = '.'):
    if nts := top_down.keys() | bottom_unary.keys():
        nts = min(nts) - 1
    new_top_down = {}
    height_cache = {}
    for node, height in height_gen(top_down, root_id):
        height_cache[node] = height # a real node
        h_children = defaultdict(dict)
        for child, info in top_down[node].children.items():
            if child in height_cache:
                ch = height_cache[child]
            else:
                ch = height_cache[child] = -1
            h_children[ch][child] = info

        new_children = {}
        p_label = top_down[node].label
        for h_level in range(min(h_children), max(h_children)):
            sub_heights = []
            sub_children = {}
            for h, c in h_children.items():
                if h <= h_level:
                    sub_heights.append(h)
                    sub_children.update(c)
            if len(sub_children) > 1:
                ftags = []
                while sub_heights:
                    ftags += h_children.pop(sub_heights.pop()).values()
                if nts is None:
                    new_node = nts = min(top_down) - 1 # node + f'{sub_suffix}{sub_start}'
                else:
                    nts -= 1; new_node = nts
                h_children[h_level + 1][new_node] = sub_suffix.join('' for x in ftags) # artificial
                new_children[new_node] = TopDown(sub_prefix + p_label, sub_children)
        if new_children:
            new_top_down.update(new_children)
            new_children = {}
            for c in h_children.values():
                new_children.update(c)
            new_top_down[node] = TopDown(p_label, new_children)
        else:
            new_top_down[node] = top_down[node]
    return new_top_down

def _new_dep(dep_head):
    return {node: TopDown(head, {node: ''}) for node, head in dep_head.items()}

def _dep_n_prefix(dep_head):
    return {f'n_{node}': f'n_{head}' if head else None for node, head in dep_head.items()}

def _dep_combine(dependency, h_node, new_node, *d_nodes):
    head = dependency.pop(h_node)
    for d_node in d_nodes: # dependants disappear
        head.children.update(dependency.pop(d_node).children)
    dependency[new_node] = head # the head keeps its dependant

def _dep_on(dependency, d_node, h_node):
    # this .label actually means .head
    return dependency[d_node].label in dependency[h_node].children

def boundary(top_down, nid):
    if nid not in top_down:
        return nid, nid
    lhs = rhs = None
    for cid in top_down[nid].children:
        l, r = boundary(top_down, cid)
        if lhs is None or l < lhs:
            lhs = l
        if rhs is None or rhs < r:
            rhs = r
    return lhs, rhs

def prepare_bit(bottom, bottom_is_bid):
    if bottom_is_bid:
        return {bid: 1 << eid for eid, bid in enumerate(bottom)}
    else:
        return {bid: 1 << eid for eid, (bid, _, _) in enumerate(bottom)}

def bit_gen(bottom, top_down, nid):
    bit_coverage = 0
    for cid in top_down[nid].children:
        if cid in bottom:
            bit_coverage |= bottom[cid] # ^= does the same for trees, not graphs
        else:
            for something in bit_gen(bottom, top_down, cid):
                bit_coverage |= something[1]
                yield something
    yield nid, bit_coverage

from utils.math_ops import bit_fanout
def gap_degree(bottom, top_down, reduce_for_nid = 0, bottom_is_bid = False):
    if not_reduce := reduce_for_nid is None:
        reduce_for_nid = 0
    if len(top_down) < 2:
        return {reduce_for_nid: 0} if not_reduce else 0
    bottom = prepare_bit(bottom, bottom_is_bid)

    gaps = {n: bit_fanout(b) - 1 for n, b in bit_gen(bottom, top_down, reduce_for_nid)}
    return gaps if not_reduce else max(gaps.values())

def bracketing(bottom, top_down, bottom_is_bid = False, excluded_labels = None):
    bottom = prepare_bit(bottom, bottom_is_bid)
    bracket_cnt = Counter()
    bracket_mul = {}
    for nid, bit in bit_gen(bottom, top_down, 0):
        label = top_down[nid].label
        if not excluded_labels or label not in excluded_labels:
            bracket_key = label, bit
            # assert bracket_key not in bracket_mul
            bracket_cnt[bracket_key] += 1
            bracket_mul[bracket_key] = len(top_down[nid].children)

    assert bit + ~bit == -1, 'Discontinuous root'
    return bracket_cnt, bracket_mul

from data.cross.dag import PathFinder
def filter_words(bottom, top_down, excluded_words, excluded_tags = None):
    remove = []
    path_finder = PathFinder(top_down, 0)
    for eid, (bid, wd, tg) in enumerate(bottom):
        if excluded_words and wd in excluded_words or excluded_tags and tg in excluded_tags:
            for path in path_finder[bid]:
                top_down[path[-2]].children.pop(bid)
            remove.append(eid)
    remove.reverse()
    for eid in remove:
        assert bottom.pop(eid)[1] in excluded_words

from utils import do_nothing
def new_word_label(bottom, top_down, *, word_fn = do_nothing, tag_fn = do_nothing, label_fn = do_nothing):
    new_bottom = [(bid, word_fn(wd), tag_fn(tg)) for bid, wd, tg in bottom]
    new_top_down = {nid: TopDown(label_fn(td.label), td.children) for nid, td in top_down.items()}
    return new_bottom, new_top_down

# def swappable_layers(layers_of_label, layers_of_right, layers_of_joint, layers_of_direc):
#     for label_layer, right_layer, joint_layer direc_layer in zip(layers_of_label, layers_of_right, layers_of_joint + [None], layers_of_direc):
#         swap_layer = []
#         this_swap = None
#         for nid, (label, right, direc) in enumerate(zip(label_layer, right_layer direc_layer)):
#             if nid and joint_layer[nid - 1] and last_right and not right:
#                 if this_swap is None:
#                     this_swap = [last_nid]
#                 this_swap.append(nid)
            
#             last_right = right
#             last_nid = nid

def _pre_proc(bottom_info, top_down, unary_join_mark = '+', dep = None):
    bu_nodes = [p_node for p_node, (_, children) in top_down.items() if len(children) == 1]
    unary = {}
    while bu_nodes:
        p_node = bu_nodes.pop()
        label, children = top_down.pop(p_node)
        node = get_sole_key(children) # prearg info lost
        unary[node] = label, p_node

    word = []
    node2tag = {}
    bottom_unary = {}
    new_bottom = []
    if dep: _dep = {}
    for node, wd, tg in bottom_info:
        word.append(wd)

        collapsed_unary_labels = []
        if dep: dep_origin = node; dep_head = dep.pop(node) if node in unary else None
        while node in unary: # bottom up node
            label, node = unary.pop(node) # shift node!
            collapsed_unary_labels.append(label)
        if collapsed_unary_labels:
            collapsed_unary_labels.reverse()
            bottom_unary[node] = unary_join_mark.join(collapsed_unary_labels)
            if dep: _dep[dep_origin] = node; dep[node] = dep_head

        new_bottom.append(node)
        node2tag[node] = tg

    for node, (label, p_node) in sorted(unary.items(), key = lambda x: x[0], reverse = True):
        td_label, children = top_down.pop(node)
        top_down[p_node] = TopDown(label + unary_join_mark + td_label, children)
    
    if dep and _dep:
        for n, h in dep.items():
            if h in _dep:
                dep[n] = _dep[h]

    return word, new_bottom, node2tag, bottom_unary

def _pre_dep(dep, node2tag):
    extra = []
    for node, bt_head in dep.items():
        if bt_head not in node2tag and bt_head:
            extra.append(node)
    media = set()
    for node in extra:
        bt_head = dep.pop(node)
        while bt_head and bt_head not in node2tag:
            media.add(bt_head)
            bt_head = dep[bt_head]
        dep[node] = bt_head
    for bt_head in media:
        dep.pop(bt_head)
            # if details:
            #     print('  '.join(n.split('_')[1]+'->'+(h.split('_')[1] if h else '*') for n, h in dep.items()))
                # if details and not bt_head:
                #     print('!:', node,' misses attachment.')
            # if details:
            #     print('  '.join(n.split('_')[1]+'->'+(h.split('_')[1] if h else '*') for n, h in dep.items()))
            #     if media:
            #         print('Removed media nodes: ' + ', '.join(media))


def _combine(parent_node, child_node, non_terminals, top_down, perserve_sub):
    if perserve_sub or child_node > 0 or non_terminals[child_node][0] not in '#_':
        top_down[parent_node].add(child_node)
        safe_label = None
    else:
        top_down[parent_node] |= top_down.pop(child_node)
        safe_label = non_terminals.pop(child_node)[1:]
        safe_label = non_terminals[parent_node].endswith(safe_label)
    return safe_label

E_SHP = 0
E_CMB = 1
E_LBL = 2

def explain_error(error_layer, error_id, sent_len):
    if error_id == E_SHP:
        error = 'Bad tensor shape'
    elif error_id == E_CMB:
        error = 'No action was taken'
    elif error_id == E_LBL:
        error = 'Combine into <nil>'
    return f'len={sent_len}, {error} at layer {error_layer}'

def bottom_trees(word, bottom_tag, layers_of_label, fallback_label, perserve_sub):
    track_nodes = []
    terminals = []
    non_terminals = {}
    top_down = defaultdict(set)
    NTS = -1
    perserve_sub &= len(layers_of_label) > 1 and len(layers_of_label[0]) > 1
    for tid, wd_tg in enumerate(zip(word, bottom_tag), 1):
        terminals.append((tid,) + wd_tg)
        label = layers_of_label[0][tid - 1]
        if perserve_sub or label[0] in '#_':
            track_nodes.append(tid)
        else:
            bottom_unary = label.split('+')
            last_node = tid
            while bottom_unary:
                non_terminals[NTS] = bottom_unary.pop()
                top_down[NTS] = set({last_node})
                last_node = NTS
                NTS -= 1
            track_nodes.append(NTS + 1)
    return NTS, tid, track_nodes, terminals, non_terminals, top_down, isinstance(fallback_label, str), None

def multi_attachment(top_down):
    children_cnt = Counter()
    for td in top_down.values():
        children_cnt += Counter(td.children.keys())
    return children_cnt

def leaves(bottom, top_down, nid):
    if nid in top_down:
        for cid in top_down[nid].children:
            yield from leaves(bottom, top_down, cid)
    else:
        yield bottom[nid - 1]

def remove_repeated_unary(top_down, nid = 0):
    if nid in top_down:
        children = top_down[nid].children
        for cid in children:
            remove_repeated_unary(top_down, cid)
        if len(children) == 1 and cid in top_down and top_down[nid].label == top_down[cid].label:
            ftag = children.pop(cid)
            top_down[nid] = top_down.pop(cid)

from random import random
def _more_sub(nid, td, rate, new_top_down, sub_prefix, nts):
    if len(td.children) > 2:
        td_in, td_out = {}, {}
        for cid, val in td.children.items():
            if random() < rate:
                td_in[cid] = val
            else:
                td_out[cid] = val
        if len(td_in) < 2 or not td_out:
            new_top_down[nid] = td
        else:
            nts -= 1
            label = td.label
            new_top_down[nid] = TopDown(label, td_out)
            new_nid = nts
            td_out[new_nid] = None
            if label[0] != sub_prefix:
                label = sub_prefix + label
            new_top_down[new_nid] = td = TopDown(label, td_in)
            if len(td_in) > 2:
                nts = _more_sub(new_nid, td, rate, new_top_down, sub_prefix, nts)
    else:
        new_top_down[nid] = td
    return nts

def add_subs(top_down, bottom_unary, rate, sub_prefix = '_'):
    new_top_down = {}
    if nts := top_down.keys() | bottom_unary.keys():
        nts = min(nts)
    for nid, td in top_down.items():
        nts = _more_sub(nid, td, rate, new_top_down, sub_prefix, nts)
    return new_top_down
    

from data.cross.art import label_only, sort_by_arity, sort_by_lhs_arity, sort_by_rhs_arity
from data.cross.art import style_1, style_2, draw_bottom, make_spans, draw_line, sbrkt_ftag
def draw_str_lines(bottom, top_down, *,
                   style_fn = style_2,
                   reverse = True,
                   sort_fn = None,
                   ftag_fn = None,
                   label_fn = label_only,
                   wrap_len = 2):
    symbols, stroke_fn = style_fn(reverse)
    word_line, tag_line, bottom_up, jobs, cursors = draw_bottom(bottom, top_down, wrap_len)
    has_ma = any(len(p) > 1 for p in bottom_up.values())
    flabel = lambda pid: label_fn(pid, top_down)
    if sort_fn is None:
        sort_fn = sort_by_lhs_arity if has_ma else sort_by_arity
    pic_width = len(tag_line)
    str_lines = [word_line, tag_line]
    while jobs:
        old_bars = cursors.copy() if callable(ftag_fn) else None
        spans, bars, jobs = make_spans(bottom_up, top_down, jobs, cursors, flabel, sort_fn, stroke_fn, has_ma, ftag_fn)
        str_lines.extend(draw_line(spans, pic_width, symbols, bars, old_bars, ftag_fn))
    if reverse:
        str_lines.reverse()
    return str_lines