from collections import namedtuple, defaultdict, Counter
from utils.param_ops import get_sole_key
TopDown = namedtuple('TopDown', 'label, children')
C_VROOT = 'VROOT'
do_nothing = lambda x: x

def descendant_path(top_down, pid, cid, path = []): # TODO: support MA
    assert cid in top_down, f'invalid pid'
    if pid in top_down:
        for c in top_down[pid].children:
            c_path = path + [c]
            if c == cid: return c_path
            c_path = descendant_path(top_down, c, cid, c_path)
            if c_path: return c_path

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

def find_labeled_on_path(top_down, root_id, target_label, child_id):
    if (path := descendant_path(top_down, root_id, child_id)) is not None:
        path.insert(0, root_id); path.reverse()
        for pid, cid in zip(path[1:], path):
            if top_down[cid].label == target_label:
                return pid, cid
    raise KeyError(f'Label \'{target_label}\' not found!')

def height_gen(top_down, root_id):
    max_height = -1
    if root_id in top_down:
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
    if not top_down:
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

def find_parent(top_down, root, cid): # TODO: support MA
    if root in top_down:
        if cid in top_down[root].children:
            return root
        for nid in top_down[root].children:
            if rid := find_parent(top_down, nid, cid):
                return rid

def bottom_up_with_root(top_down, nids = {0}, include_bottom = True):
    bottom_up = defaultdict(set)
    while nids:
        cids = set()
        for nid in nids:
            if nid in top_down:
                children = top_down[nid].children
                cids.update(children)
                for cid in children:
                    if include_bottom or cid in top_down:
                        bottom_up[cid].add(nid)
        nids = cids
    return bottom_up

def filter_words(bottom, top_down, excluded_words, excluded_tags = None):
    remove = []
    for eid, (bid, wd, tg) in enumerate(bottom):
        if excluded_words and wd in excluded_words or excluded_tags and tg in excluded_tags:
            top_down[find_parent(top_down, 0, bid)].children.pop(bid)
            remove.append(eid)
    remove.reverse()
    for eid in remove:
        assert bottom.pop(eid)[1] in excluded_words

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

def bottom_trees(word, bottom_tag, layers_of_label, fall_back_root_label, perserve_sub):
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
    return NTS, tid, track_nodes, terminals, non_terminals, top_down, isinstance(fall_back_root_label, str), None

def multi_attachment(top_down):
    children_cnt = Counter()
    for td in top_down.values():
        children_cnt += Counter(td.children.keys())
    return {k:v for k,v in children_cnt.items() if v > 1}

def leaves(bottom, top_down, nid):
    if nid in top_down:
        for cid in top_down[nid].children:
            yield from leaves(bottom, top_down, cid)
    else:
        yield bottom[nid - 1]

from data.cross.art import label_only, sort_by_num_children, sort_by_num_children
from data.cross.art import style_1, style_2, draw_bottom, make_spans, draw_line
def draw_str_lines(bottom, top_down,
                   reverse = True,
                   label_fn = label_only,
                   style_fn = style_2,
                   priority = sort_by_num_children,
                   ftag_fn  = None,
                   wrap_len = 1):
    symbols, stroke_fn = style_fn(reverse)
    word_line, tag_line, bottom_up, jobs, cursors = draw_bottom(bottom, top_down, wrap_len)
    pic_width = len(tag_line)
    str_lines = [word_line, tag_line]
    while jobs:
        old_bars = cursors.copy() if callable(ftag_fn) else None
        spans, bars, jobs = make_spans(bottom_up, top_down, jobs, cursors, label_fn, priority, stroke_fn)
        str_lines.extend(draw_line(spans, pic_width, symbols, bars, old_bars, ftag_fn))
    if reverse:
        str_lines.reverse()
    return str_lines

def remove_repeated_unary(top_down, nid = 0):
    if nid in top_down:
        children = top_down[nid].children
        for cid in children:
            remove_repeated_unary(top_down, cid)
        if len(children) == 1 and cid in top_down and top_down[nid].label == top_down[cid].label:
            ftag = children.pop(cid)
            top_down[nid] = top_down.pop(cid)