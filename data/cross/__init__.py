from collections import namedtuple, defaultdict, Counter
from utils.param_ops import get_sole_key
TopDown = namedtuple('TopDown', 'label, children')

def has_multiple(gen):
    count = 0
    for state in gen:
        count += state
        if count > 1:
            return True
    return False

# def list_swap(bottom, lhs, rhs):
#     bottom[lhs], bottom[rhs] = bottom[rhs], bottom[lhs]
from utils.shell_io import byte_style

def _read_graph(graph, make_up_for_no_nt = True):
    bottom_up = {}
    top_down = {}
    single_attachment = set()
    terminals, non_terminals = graph[0]
    bottom = [(t.get('id'), t.get('word'), t.get('pos')) for t in terminals]

    for nt in non_terminals:
        p_node = nt.get('id')
        label = nt.get('cat')
        children = {}

        for edge in nt:
            if edge.tag == 'secedge':
                continue
            node = edge.get('idref')
            children[node] = edge.get('label')
            assert node not in single_attachment, 'multi-attachment'
            single_attachment.add(node)
            top_down[p_node] = TopDown(label, children)
            bottom_up[node] = p_node

    if top_down:
        while p_node in bottom_up: # to root
            p_node = bottom_up[p_node]
    elif make_up_for_no_nt:
        p_node = graph.get('id') + '_VROOT'
        top_down[p_node] = TopDown('VROOT', {bid: '--' for bid, _, _ in bottom})

    validate(bottom, top_down, p_node)
    return bottom, top_down, p_node

_CMD_TAG = 0
_CMD_BOL = 1
_CMD_EOL = 2
E_DISCO = '*T*', '*ICH*', '*EXP*', '*RNR*'
from data.delta import preproc_cnf

def remove_irrelevant_trace(tree):
    bottom = list(enumerate(tree.pos()))
    bottom.reverse()
    for bid, (word, tag) in bottom:
        is_not_trace   = tag != '-NONE-'
        is_disco_trace = any(word.startswith(tc) for tc in E_DISCO)
        if is_not_trace or is_disco_trace:
            continue
        
        tag_path = tree.leaf_treeposition(bid)[:-1]
        syn_path = tag_path[:-1] # leaf must be unary
        if len(tree[syn_path]) > 1: # more than one child
            # NP (SBAR) (-NONE- *-1)
            syn_path = tag_path
        else: # NP -NONE- *-1
            while syn_path and len(tree[syn_path[:-1]]) == 1:
                syn_path = syn_path[:-1]
        del tree[syn_path]

def remove_eq(label):
    pos = label.find('=')
    if pos < 0:
        return label
    return label[:pos]

def _preorder(tree):
    if tree.height() < 3:
        assert len(tree) == 1
        word = tree[0]
        if '\\' in word: # single \ in nltk.cp.tb
            word = word.replace('\\', '')
        elif word == '-LRB-':
            word = '('
        elif word == '-RRB-':
            word = ')'
        yield _CMD_TAG
        yield word, tree.label()
    else:
        for child in tree:
            yield from _preorder(child)
        yield _CMD_BOL
        for child in reversed(tree):
            yield remove_eq(child.label())
        yield _CMD_EOL
        yield remove_eq(tree.label())

def boundaries(top_down, nid):
    if nid not in top_down:
        return nid, nid
    nids = [nid]
    cids = []
    coverage = []
    while nids:
        for nid in nids:
            for cid in top_down[nid].children:
                if cid in top_down:
                    cids.append(cid)
                else:
                    assert cid < 500
                    coverage.append(cid)
        nids = cids
        cids = []
    return min(coverage), max(coverage)

def is_a_child(top_down, pid, cid):
    if pid < 500:
        return False
    if cid in top_down[pid].children:
        return True
    cids = []
    nids = list(c for c in top_down[pid].children if c in top_down)
    while nids:
        for nid in nids:
            if cid in top_down[nid].children:
                return True
            cids.extend(top_down[nid].children)
        nids = [c for c in cids if c in top_down]
        cids = []
    return False

def __validate(being_bids, to_be_bids, top_down, checked_nids):
    redundant_bids = being_bids - to_be_bids
    redundant_nids = top_down.keys() - checked_nids
    if to_be_bids ^ being_bids:
        if to_be_bids - being_bids:
            msg = f'Lacking bids: {to_be_bids - being_bids}'
        else:
            msg = f'Redundant bids: {redundant_bids}'
        raise ValueError(msg)
    elif redundant_nids:
        for nid in redundant_nids:
            _, children = top_down.pop(nid)
            safe = True
            for cid in children:
                if cid < 500:
                    safe &= cid not in being_bids
                else:
                    safe &= cid in redundant_nids
                if not safe:
                    break
            if not safe:
                raise ValueError(f'Redundant nids: {redundant_nids}')

    if checked_nids ^ top_down.keys():
        if checked_nids - top_down.keys():
            msg = f'Should not happen here {checked_nids - top_down.keys()}'
        else:
            msg = f'Redundant nids: {redundant_nids}'
        raise ValueError(msg)

def validate(bottom_info, top_down, root_id):
    cids = set()
    nids = [root_id]
    to_be_bids = set()
    being_bids = set(bid for bid, _, _ in bottom_info)
    checked_nids = set()
    while nids:
        for nid in nids:
            if nid in being_bids:
                to_be_bids.add(nid)
            elif nid not in top_down:
                raise ValueError(f'nid({nid}) not in top_down[\'{set(top_down)}\']')
            checked_nids.add(nid)
            for cid in top_down[nid].children:
                if cid in being_bids:
                    to_be_bids.add(cid)
                else:
                    cids.add(cid)
        nids = cids
        cids = set()
    __validate(being_bids, to_be_bids, top_down, checked_nids)

def validate_and_maintain(bottom_info, top_down, root_id, trace_dst):
    cids = set()
    nids = [root_id]
    bottom_up = {}
    to_be_bids = set()
    being_bids = set(bid for bid, _, _ in bottom_info)
    checked_nids = set()
    while nids:
        for nid in nids:
            if nid < 500:
                to_be_bids.add(nid)
            elif nid not in top_down:
                raise ValueError(f'nid({nid}) not in top_down[\'{set(top_down)}\']')
            checked_nids.add(nid)
            for cid in top_down[nid].children:
                if cid < 500:
                    to_be_bids.add(cid)
                else:
                    cids.add(cid)
                bottom_up[cid] = nid
        nids = cids
        cids = set()
    remove_bids = (bid for bid, _, tag in bottom_info if tag == '-NONE-')
    remove_bids = sorted(remove_bids, reverse = True)
    remove_cids = [s_pid for s_pid in top_down if s_pid != root_id and not top_down[s_pid].children]
    for cid in remove_cids + remove_bids:
        if cid in remove_bids:
            bid = cid - sum(td.bid < cid for td in trace_dst)
            assert bottom_info.pop(bid)[2] == '-NONE-'
            to_be_bids.remove(cid)
            being_bids.remove(cid)
        else:
            top_down.pop(cid)
            checked_nids.remove(cid)
        pid = bottom_up.pop(cid)
        top_down[pid].children.pop(cid)
        while not top_down[pid].children: # empty again
            top_down.pop(pid)
            cid = pid
            checked_nids.remove(cid)
            pid = bottom_up.pop(cid)
            top_down[pid].children.pop(cid)
    __validate(being_bids, to_be_bids, top_down, checked_nids)


TraceSrc = namedtuple('TraceSrc', 'pid, cid, lhs, rhs')
TraceDst = namedtuple('TraceDst', 'typ, tid, pid, cid, bid')
from nltk.tree import Tree

def trace_dst_gen(trace_src, trace_dst):
    # all trace_dst should be projected
    for tid in trace_dst.keys() - trace_src.keys():
        trace_dst.pop(tid) # 1 in nltk treebank

    # select nearest attachment
    for tid, tds in trace_dst.items():
        num_tds = len(tds)
        if num_tds > 1:
            _, _, lhs, rhs = trace_src[tid]
            distances = {}
            for ti, td in enumerate(tds):
                d_bid = td.bid
                if td.tid == tid:
                    lh = max(lhs - d_bid, 0)
                    rh = max(d_bid - rhs, 0)
                    distances[ti] = max(lh, rh)
            yield tds[min(distances, key = distances.get)]
        else:
            yield tds[0]

def _read_dpenn(tree, convert_id_to_str = True):
    bottom = []
    top_down = {}
    pd_args = {}
    trace_src = {}
    trace_dst = defaultdict(list)
    stack = defaultdict(set)
    remove_irrelevant_trace(tree)
    tree = Tree('VROOT', [tree])
    for item in _preorder(tree):
        if isinstance(item, int):
            status = item
            if status == _CMD_BOL:
                nid = 500 + len(top_down)
                top_down[nid] = []
                stack['__CURRENT__'].add(nid)
        elif status == _CMD_TAG:
            wd, tg = item
            nid = len(bottom)
            bottom.append((nid, wd, tg))
            stack[tg].add(nid)
            
            if wd[0] == '*' and wd[-1].isdigit() and '-' in wd[1:-1]:
                _args = wd.split('-')
                if _args[0] in E_DISCO:
                    tid = _args.pop()
                    tp_ = '-'.join(_args)
                    assert tg == '-NONE-'
                    trace_dst[nid] = tp_, tid
        elif status == _CMD_BOL:
            # item is a tag or a label
            cnid = max(stack[item])
            stack[item].remove(cnid)
            if not stack[item]:
                stack.pop(item)
            top_down[nid].append(cnid)
        elif status == _CMD_EOL:
            # item is the parent label
            stack[item] |= stack.pop('__CURRENT__')

            if '-' in item:
                _args = item.split('-')
                item = _args.pop(0)
                if _args[-1].isdigit():
                    trace_src[nid] = _args.pop()
                if _args:
                    pd_args[nid] = '-'.join(_args)

            children = {}
            for cnid in top_down[nid]:
                children[cnid] = pd_args.pop(cnid, '')

                if cnid in trace_src:
                    lhs, rhs = boundaries(top_down, cnid)
                    tid = trace_src.pop(cnid)
                    if tid in trace_src:
                        was_wh_movement = top_down[trace_src[tid].cid].label.startswith('WH')
                        # trace_src[tid].lhs 
                        if not was_wh_movement: # wh-movement has the priority
                            trace_src[tid] = TraceSrc(nid, cnid, lhs, rhs)
                    else:
                        trace_src[tid] = TraceSrc(nid, cnid, lhs, rhs)

                if cnid in trace_dst:
                    ty_id = trace_dst.pop(cnid)
                    if len(ty_id) == 2:
                        trace_dst[nid] = ty_id + (cnid,)
                    elif len(ty_id) == 3:
                        typ, tid, bid = ty_id
                        trace_dst[tid].append(TraceDst(typ, tid, nid, cnid, bid))

            top_down[nid] = TopDown(item, children)
    assert not pd_args or nid in pd_args
    assert len(stack) == 1
    assert nid in stack['VROOT']

    if trace_dst:
        trace_dst = trace_dst_gen(trace_src, trace_dst)
        trace_dst = sorted(trace_dst, key = lambda td: td.bid, reverse = True)
    else:
        trace_src = [] # change type

    # cross trace along the bottom (ordered and reversed for bottom.pop(i) stability)
    history = {}
    for _, tid, d_pid, d_cid, d_bid in trace_dst:
        s_pid, s_cid, lhs, rhs = trace_src.pop(tid)
        d_pid = history.pop(d_cid, d_pid)
        s_ftag = top_down[s_pid].children.pop(s_cid)
        d_ftag = top_down[d_pid].children.pop(d_cid)
        v_bid, v_wd, v_tg = bottom.pop(d_bid)
        assert v_wd.endswith(tid)
        assert (d_bid, '-NONE-') == (v_bid, v_tg)
        if s_ftag and d_ftag:
            ftag = s_ftag if s_ftag == d_ftag else (s_ftag + '-' + d_ftag)
        else:
            ftag = s_ftag or d_ftag
        top_down[d_pid].children[s_cid] = ftag
        history[s_cid] = d_pid
        if lhs <= d_bid <= rhs:
            for s_ccid in top_down[s_cid].children:
                if is_a_child(top_down, s_ccid, d_pid):
                    break
            ftag = top_down[s_cid].children.pop(s_ccid)
            top_down[s_pid].children[s_ccid] = ftag

    validate_and_maintain(bottom, top_down, nid, trace_dst)
    vroot = top_down.pop(nid)
    assert vroot.label == 'VROOT'
    root_id = get_sole_key(vroot.children)

    if convert_id_to_str:
        new_bottom = [(f'n_{bid}', w, t) for bid, w, t in bottom]
        new_top_down = {}
        for nid, td in top_down.items():
            children = {}
            for cid, ftag in td.children.items():
                children[f'n_{cid}'] = ftag
            new_top_down[f'n_{nid}'] = TopDown(td.label, children)
        return new_bottom, new_top_down, f'n_{root_id}'

    return bottom, top_down, root_id

from utils.math_ops import bit_fanout
def gap_degree(bottom, top_down, nid, bottom_is_bid = True):
    finally_return = True
    if isinstance(bottom, dict):
        if nid in bottom:
            return {nid: 0}, bottom[nid]
        finally_return = False
    else:
        if not top_down:
            return {0: 1}
        if bottom_is_bid:
            bottom = {bid: 1 << eid for eid, bid in enumerate(bottom)}
        else:
            bottom = {bid: 1 << eid for eid, (bid, _, _) in enumerate(bottom)}

    gap_cnt = Counter()
    bit_coverage = 0 # set()
    for cid in top_down[nid].children:
        child_gaps, child_coverage = gap_degree(bottom, top_down, cid)
        gap_cnt.update(child_gaps)
        bit_coverage ^= child_coverage

    if finally_return: # or nid.endswith('_VROOT'):
        assert bit_fanout(bit_coverage) == 1
        return gap_cnt

    gap_cnt[bit_fanout(bit_coverage) - 1] += 1
    return gap_cnt, bit_coverage

def bracketing(bottom, top_down, nid, bottom_is_bid = False,
               unlabel = None,
               excluded_labels = None,
               equal_labels = None):
    final_check = False
    if isinstance(bottom, dict):
        if nid in bottom:
            return bottom[nid]
    else:
        if bottom_is_bid:
            bottom = {bid: 1 << eid for eid, bid in enumerate(bottom)}
        else:
            bottom = {bid: 1 << eid for eid, (bid, _, _) in enumerate(bottom)}
        if not top_down:
            return sum(bottom.values())
        final_check = True

    bit_coverage = 0
    bracket_cnt = Counter()
    for cid in top_down[nid].children:
        something = bracketing(bottom, top_down, cid, unlabel = unlabel)
        if isinstance(something, int): # from a terminal
            bit_coverage ^= something
        else: # from a non-terminal
            bracket_cnt.update(something)
            for _, child_coverage in something.keys():
                bit_coverage |= child_coverage

    if unlabel is None:
        label = top_down[nid].label
        if equal_labels:
            label = equal_labels.get(label, label)
    else:
        label = unlabel
    if not excluded_labels or label not in excluded_labels:
        bracket_cnt[(label, bit_coverage)] += 1
    if final_check:
        assert bit_coverage + ~bit_coverage == -1
    return bracket_cnt

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
        

def _pre_proc(bottom_info, top_down, unary_join_mark = '+'):
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
    for node, wd, tg in bottom_info:
        word.append(wd)

        collapsed_label = ''
        while node in unary: # bottom up node
            label, node = unary.pop(node) # shift node!
            collapsed_label += unary_join_mark + label
        if collapsed_label:
            bottom_unary[node] = collapsed_label[1:]

        new_bottom.append(node)
        node2tag[node] = tg

    for node, (label, p_node) in sorted(unary.items(), key = lambda x: x[0]): # collapse top_down unary branches
        td_label, children = top_down.pop(node)
        top_down[p_node] = TopDown(label + unary_join_mark + td_label, children)

    return word, new_bottom, node2tag, bottom_unary

def _layer_base(bottom, top_down, completed_nodes, some_or_all, lean_joint, cnf_right, sub_suffix):
    bottom_up = {}
    right_hash = {}
    swappable_locations = []
    joint_rset = []
    bottom_flag = [True for _ in bottom]
    for p_node, td in top_down.items():
        if p_node in completed_nodes or not some_or_all(node in bottom for node in td.children):
            continue
        
        location = []
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                location_children.append((nid, node))
        swappable_locations.append(location)

        num_children = len(location_children)
        for cid, (nid, node) in enumerate(location_children):
            if cid == 0:
                right = True
            else:
                if cid == num_children - 1:
                    right = False
                elif cnf_right is None:
                    right = (cid / num_children) < 0.5
                else:
                    right = cnf_right
                if last_nid == nid - 1: # locality
                    assert last_right or not right # coherent
                    if not lean_joint:
                        joint_rset.append(nid)
                    if last_right and not right: # agreeable_orients
                        if lean_joint:
                            joint_rset.append(nid)
                        group = td.children
                        if len(group) == 2:
                            new_node = p_node
                            completed_nodes.add(p_node)
                        else:
                            sub_id = 0
                            new_node = str(p_node)
                            while new_node + f'{sub_suffix}{sub_id}' in group:
                                sub_id += 1
                            new_node += f'{sub_suffix}{sub_id}'
                        bottom_up[node] = new_node
                        group[new_node] = group.pop(last_node) + '.' + group.pop(node)
                        bottom_flag[last_nid] = False
                    bottom_flag[nid] = False
                    
            right_hash[node] = last_right = right
            last_nid = nid
            last_node = node
    return right_hash, joint_rset, swappable_locations, bottom_up

def _layer_output(bottom,
                  bottom_up,
                  top_down,
                  node2tag,
                  bottom_unary,
                  right_hash,
                  joint_rset,
                  swap_right_priority,
                  cnf_right,
                  sub_prefix,
                  sub_suffix,
                  pos_prefix):
    last_right = True
    last_direc = False
    swap_distance = len(bottom)
    new_bottom = []
    right_layer = [] 
    direc_layer = []
    joint_layer = []
    label_layer = []
    for nid, node in enumerate(bottom):
        # if bottom[0] == '_s8511_500':
        #     import pdb; pdb.set_trace()
        if node in right_hash:
            right = right_hash[node]
            if last_right and not right:
                if nid in joint_rset: # [>≥]j<
                    assert new_bottom.pop() == last_node
                    new_bottom.append(bottom_up.pop(node))
                else: # active swap [>≥]s<
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
            elif not last_direc and not right: # passive swap
                if swap_distance != 1: # o≥s<
                    right_layer[-1] = True
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                elif swap_right_priority: # >s[≥]s< restore lhs
                    right_layer[-1] = True
                    prev_node = new_bottom.pop(-2)
                    new_bottom += [node, prev_node]
                    swap_distance = 0
                else: # ??
                    new_bottom.append(node)
            else: # >
                new_bottom.append(node)
            directional = True
        else:
            if last_direc:
                if last_right: # passive swap >s≤
                    right = False
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else: # [<]o[≤≥]
                    right = cnf_right
                    new_bottom.append(node)
            else: # [≤≥]o[≤≥] TODO undetermined right
                right = cnf_right
                new_bottom.append(node)
            directional = False
        right_layer.append(right)
        direc_layer.append(directional)
        last_direc = directional
        last_right = right
        last_node = node
        swap_distance += 1
        if nid > 0:
            is_joint = nid in joint_rset
            joint_layer.append(is_joint)
        
        if sub_suffix in node:
            label_layer.append(sub_prefix + top_down[node.split('.')[0]].label)
        elif node in bottom_unary:
            label_layer.append(bottom_unary[node])
        elif node in node2tag:
            label_layer.append(pos_prefix + node2tag[node])
        else:
            label_layer.append(top_down[node].label)

    assert len(right_layer) == len(label_layer) == len(direc_layer)
    assert len(right_layer) - 1 == len(joint_layer)
    return new_bottom, right_layer, joint_layer, label_layer, direc_layer


def cross_signals(bottom, node2tag, bottom_unary, top_down, cnf_right,
                  aggressive = True,
                  swap_right_priority = None,
                  lean_joint = False,
                  sub_prefix = '_',
                  pos_prefix = '#'):
    if swap_right_priority is None:
        swap_right_priority = not cnf_right

    some_or_all = has_multiple if aggressive else all
    sub_suffix = '.'

    layers_of_right = []
    layers_of_joint = []
    layers_of_label = []
    layers_of_direc = []
    layers_of_swaps = []
    completed_nodes = set()
    while len(bottom) > 1:
        (right_hash, joint_rset, swappable_locations,
         bottom_up) = _layer_base(bottom,
                                  top_down,
                                  completed_nodes,
                                  some_or_all,
                                  lean_joint,
                                  cnf_right,
                                  sub_suffix)

        (new_bottom, right_layer, joint_layer, label_layer,
         direc_layer) = _layer_output(bottom,
                                      bottom_up,
                                      top_down,
                                      node2tag,
                                      bottom_unary,
                                      right_hash,
                                      joint_rset,
                                      swap_right_priority,
                                      cnf_right,
                                      sub_prefix,
                                      sub_suffix,
                                      pos_prefix)
        if new_bottom == bottom:
            raise ValueError('should be different', bottom, top_down, bottom_unary, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc)
        bottom = new_bottom
        layers_of_right.append(right_layer)
        layers_of_joint.append(joint_layer)
        layers_of_label.append(label_layer)
        layers_of_direc.append(direc_layer)
        layers_of_swaps.append(swappable_locations)

    if top_down:
        layers_of_right.append([cnf_right])
        layers_of_direc.append([False])
        layers_of_label.append([top_down[bottom.pop()].label])
    return layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, layers_of_swaps

from utils.types import E_CNF, O_RGT
from copy import deepcopy
def read_tiger_graph(graph):
    bottom_info, top_down, root_id = _read_graph(graph)
    # ret_top_down = deepcopy(top_down)
    word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
    bottom_tag = [node2tag[t] for t in bottom]
    gap = gap_degree(bottom, top_down, root_id)
    cnf_layers = {}
    for cnf_factor in E_CNF:
        new_top_down = deepcopy(top_down) if cnf_factor != O_RGT else top_down
        cnf_layers[cnf_factor] = cross_signals(bottom, node2tag, bottom_unary, new_top_down, cnf_factor == O_RGT)
    return word, bottom_tag, cnf_layers, gap #, bottom_info, ret_top_down, root_id

def read_disco_penn(tree):
    bottom_info, top_down, root_id = _read_dpenn(tree)
    word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
    bottom_tag = [node2tag[t] for t in bottom]
    gap = gap_degree(bottom, top_down, root_id)
    cnf_layers = {}
    for cnf_factor in E_CNF:
        new_top_down = deepcopy(top_down) if cnf_factor != O_RGT else top_down
        cnf_layers[cnf_factor] = cross_signals(bottom, node2tag, bottom_unary, new_top_down, cnf_factor == O_RGT)
    return word, bottom_tag, cnf_layers, gap

def targets(right_layer, joint_layer):
    targets = [1 for _ in right_layer]
    tar_len = len(targets)
    for r0id, (r0, r1, jc) in enumerate(zip(right_layer, right_layer[1:], joint_layer)):
        if r0 and not r1:
            if not jc:
                targets[r0id] += 1
                targets[r0id + 1] -= 2
                if r0id + 2 < tar_len:
                    targets[r0id + 2] += 1
    return targets

E_SHP = 0
E_CMB = 1

def explain_error(error_layer, error_id, sent_len):
    if error_id == E_SHP:
        error = 'Bad tensor shape'
    elif error_id == E_CMB:
        error = 'No action was taken'
    return f'len={sent_len}, {error} at layer {error_layer}'

def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_right,
               layers_of_joint,
               layers_of_direc,
               fall_back_root_label = None,
               perserve_sub         = False):
    track_nodes = []
    terminals = []
    non_terminals = {}
    top_down = defaultdict(set)
    track_fall_back = isinstance(fall_back_root_label, str)
    error_layer_id = None
    NTS = 500
    for tid, wd_tg in enumerate(zip(word, bottom_tag)):
        terminals.append((tid,) + wd_tg)
        if perserve_sub or layers_of_label[0][tid][0] in '#_':
            track_nodes.append(tid)
        else:
            bottom_unary = layers_of_label[0][tid].split('+')
            last_node = tid
            while bottom_unary:
                non_terminals[NTS] = bottom_unary.pop()
                top_down[NTS] = set({last_node})
                last_node = NTS
                NTS += 1
            track_nodes.append(NTS - 1)
    bottom_len = tid + 1
            
    non_terminal_start = NTS
    def _combine(child_node, parent_node, non_terminals, top_down):
        if perserve_sub or child_node < NTS or non_terminals[child_node][0] not in '#_':
            top_down[parent_node].add(child_node)
            safe_label = None
        else:
            top_down[parent_node] |= top_down.pop(child_node)
            safe_label = non_terminals.pop(child_node)[1:]
            safe_label = non_terminals[parent_node].endswith(safe_label)
        return safe_label

    for lid, (lr, lj, ld) in enumerate(zip(layers_of_right, layers_of_joint, layers_of_direc)):
        # if not (len(lr) == len(ld) == len(lj) + 1 == len(track_nodes)):
        #     print(f'{len(lr)} == {len(ld)} == {len(lj)} + 1 == {len(lr)}')
        #     import pdb; pdb.set_trace()
        # assert len(lr) == len(ld) == len(track_nodes), f'{len(lr)} == {len(ld)} == {len(lr)} @ {lid}'
        # if lj:
        #     assert len(lj) + 1 == len(track_nodes), f'{len(lj)} + 1 == {len(lr)} @ {lid}'
        snapshot_track_nodes = track_nodes.copy()
        offset = 1
        for nid, (right, direc) in enumerate(zip(lr, ld)):
            lhs_nid = nid - offset
            if nid == 0:
                pass
            elif len(lj) == 0:
                error_layer_id = lid, E_SHP, bottom_len
                break # should be the final | unknown action in the model
            elif lj[nid - 1]: # joint
                if last_right and not right:
                    # >j<
                    lhs_node = track_nodes.pop(lhs_nid)
                    rhs_node = track_nodes.pop(lhs_nid)
                    try:
                        if not (lhs_nid < len(layers_of_label[lid + 1])):
                            import pdb; pdb.set_trace()
                            break # an error
                    except:
                        error_layer_id = lid, E_SHP, bottom_len # TODO change err_type
                        break
                    labels = layers_of_label[lid + 1][lhs_nid]
                    labels = [labels] if perserve_sub or labels[0] in '#_' else labels.split('+')
                    non_terminals[non_terminal_start] = labels.pop()
                    _combine(lhs_node, non_terminal_start, non_terminals, top_down) # TODO: safe_label validate
                    _combine(rhs_node, non_terminal_start, non_terminals, top_down)
                    while labels: # unary
                        non_terminal_start += 1
                        non_terminals[non_terminal_start] = labels.pop()
                        top_down[non_terminal_start] = set({non_terminal_start - 1})
                    track_nodes.insert(lhs_nid, non_terminal_start)
                    non_terminal_start += 1
                    offset += 1
            elif last_right and not right and (last_direc or direc): # cross (swap)
                # 1: >[<≤]
                # 2: [>≥]<
                # last_right and not right: # cross (swap)
                #layers_of_label[lid][nid-1:nid+1][::-1] == layers_of_label[lid+1][lhs_nid:lhs_nid+2] and\
                # 1: [≥>][<≤]
                rhs_nid = lhs_nid + 1
                # import pdb; pdb.set_trace()
                track_nodes[lhs_nid], track_nodes[rhs_nid] = track_nodes[rhs_nid], track_nodes[lhs_nid]

            last_right = right
            last_direc = direc

        # if isinstance(fall_back_root_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and snapshot_track_nodes == track_nodes and track_fall_back: # no action is taken
            if error_layer_id is None:
                error_layer_id = lid, E_CMB, bottom_len
            for pid in track_nodes:
                if pid in non_terminals and non_terminals[pid][0] in '_#':
                    non_terminals.pop(pid)
                    top_down[non_terminal_start].update(top_down.pop(pid))
                else:
                    top_down[non_terminal_start].add(pid)
            non_terminals[non_terminal_start] = fall_back_root_label
            non_terminal_start += 1
            break

    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, top_down.pop(nid))

    return terminals, top_down, non_terminal_start - 1, error_layer_id

from data.delta import get_logits as _get_logits
def xlogit_gen(label_layer, right_layer, direc_layer, joint_layer, current_joints, next_joints):
    for nid, (label, right, direc) in enumerate(zip(label_layer, right_layer, direc_layer)):
        is_phrase = label[0] not in '#_'
        is_joint = current_joints and current_joints[nid]
        if nid == 0:
            if not right and not direc:
                next_joints.append(False)
        elif last_right or not right:
            if joint_layer:
                jnt = joint_layer[nid - 1] # and last_right and last_direc and not right and direc
                if not jnt and (last_right and not right):# or last_right and not direc or not last_direc and not right):
                    next_joints.append(False)
            else:
                jnt = True # final layer
            next_joints.append(jnt)
        yield _get_logits(right, direc, is_joint, is_phrase, False, False)
        last_right = right
        last_direc = direc
    if right and not direc:
        next_joints.append(False)

def zip_to_logit(layers_of_label, layers_of_right, layers_of_joint, layers_of_direc):
    nj = []
    cindex = []
    labels = []
    xtypes = []
    for ll, lr, lj, ld in zip(layers_of_label, layers_of_right, layers_of_joint + [None], layers_of_direc):
        if nj:
            cj = nj
            nj = []
            assert len(cj) == len(ll), f'{len(cj)} != {len(ll)}'
        else:
            cj = None
        cindex.append(len(ll))
        labels.extend(ll)
        xtypes.extend(xlogit_gen(ll, lr, ld, lj, cj, nj))
    return cindex, labels, xtypes

def zip_swaps(lsp):
    swapbl = ''
    num_layers = len(lsp)
    for lid, group_layer in enumerate(lsp):
        num_groups = len(group_layer)
        for gid, group in enumerate(group_layer):
            if gid == num_groups - 1:
                if lid == num_layers - 1:
                    swapbl += ','.join(str(x) for x in group)
                else:
                    swapbl += ','.join(str(x) for x in group) + '|'
            else:
                swapbl += ','.join(str(x) for x in group) + ';'
    return swapbl

def unzip_swaps(swapbl, offset = 0):
    lsp = []
    for layer in swapbl.split('|'):
        groups = []
        for group in layer.split(';'):
            group_idx = array([int(x) + offset for x in group.split(',')])
            group_ctn = array(group_idx)
            groups.append((group_idx, group_ctn))
        lsp.append(tuple(groups))
    return lsp

from data.trapezoid import trapezoid_to_layers
from numpy import asarray, array
from data.delta import get_rgt, get_jnt, get_dir
def unzip_xlogit(cindex, xtypes): #, the_joints, the_labels):
    xtypes = asarray(xtypes)
    rights = get_rgt(xtypes)
    joints = get_jnt(xtypes)
    direcs = get_dir(xtypes)
    joints = trapezoid_to_layers(joints, cindex, cindex, big_endian = False)
    layers_of_direc = trapezoid_to_layers(direcs, cindex, cindex, big_endian = False)
    layers_of_right = trapezoid_to_layers(rights, cindex, cindex, big_endian = False)
    layers_of_joint = []
    count = 0
    for lower_rights, lower_direcs, upper_joints in zip(layers_of_right, layers_of_direc, joints[1:]):
        joint_layer = []
        upper_joints = list(upper_joints)
        upper_joints.reverse() # pop() should be like dequeue
        for rid, (right, direc) in enumerate(zip(lower_rights, lower_direcs)):
            if rid:
                if not last_right and right and rid > 1 and not cnf_left_starter: # <≤≥>
                    joint_layer.append(False)
                else:
                    joint = upper_joints.pop()
                    joint_layer.append(joint)
                    if last_right and not right and not joint: # a swap and (last_direc or direc)
                        assert not upper_joints.pop(), (count, rid)
                if cnf_left_starter and right:
                    cnf_left_starter = False
            else:
                cnf_left_starter = not right
            last_right = right
            last_direc = direc
        if upper_joints: # cnf_right_ender
            assert not upper_joints.pop()
        layers_of_joint.append(joint_layer)
        count += 1
              
                # if joint_layer[-1] != tj[len(joint_layer) - 1]:
                #     print(' '.join(path))
                #     import pdb; pdb.set_trace()
    # for aa, bb in zip(ld, layers_of_direc):lj, lr, ld:, lj, layers_of_label, ld, tj, tl, td
    #     assert list(bb) == aa, f'{aa} != {bb}'  
    #     c = ' '.join(str(int(x)) for x in upper_joints) + ']\nTj['
    #     a = ' '.join(str(int(x)) for x in tj) + ']\nLj['
    #     b = ' '.join(str(int(x)) for x in joint_layer) + ']\nTd['
    #     e = ' '.join(str(int(x)) for x in td) + ']\nLb['
    #     d = ' '.join((l + f'≥>'[d]) if r else (f'≤<'[d] + l) for l,d,r in zip(tl, td, lower_rights))
    #     assert len(joint_layer) == len(lower_rights) - 1, f'{len(joint_layer)} != {len(lower_rights)} - 1\n' + d
    #     if joint_layer != tj:
    #         print(' '.join(path))
    #         import pdb; pdb.set_trace()
    #     assert joint_layer == tj, '\nUj[' + c + a + b + e + d + ']\n' + f'{path}'
    return layers_of_right, layers_of_joint, layers_of_direc

class SpanTale:
    def __init__(self):
        self._spans = {}

    def add(self, start, end = None):
        self._spans[start] = end

    def overlapped(self, start, end = None):
        for s_start, s_end in self._spans.items():
            if end is None:
                if s_end is None:
                    if s_start == start:
                        return True
                elif s_start <= start <= s_end:
                    return True
            elif start <= s_start <= end:
                return True
            elif s_end is not None and s_start <= start <= s_end:
                return True
        return False

    def __str__(self):
        if not self._spans:
            return 'Nothing'
        last_end = 0
        blocks = ''
        for start, end in sorted(self._spans.items(), key = lambda x: x[0]):
            if end is None:
                end = start
            blocks += '_' * (start - last_end)
            blocks += '&' * (end - start + 1)
            last_end = end + 1
        blocks += f':{last_end - 1}'
        return blocks


def draw_str_lines(bottom, top_down, reverse = True, root_stamp = ''):
    if reverse:
        LC, MC, RC, MP = '┌┬┐┴'
    else:
        LC, MC, RC, MP = '└┴┘┬'
    bottom_up = {}
    for pid, td in top_down.items():
        for cid in td.children:
            bottom_up[cid] = pid
    while pid in bottom_up:
        pid = bottom_up[pid]
    root_id = pid
    str_lines = []
    word_line = ''
    tag_line = ''
    start_bars = set()
    next_top_down = defaultdict(list)
    for bid, word, tag in bottom:
        unit_len = max(len(word), len(tag)) + 2
        word_line += word.center(unit_len)
        tag_line  +=  tag.center(unit_len)
        mid_pos = len(word_line) - round(unit_len // 2)
        start_bars.add(mid_pos)
        next_top_down[bottom_up[bid]].append((bid, mid_pos))
    pic_width = len(word_line)
    str_lines.append(word_line)
    str_lines.append(tag_line)
    # print(word_line)
    # print(tag_line)
    while next_top_down:
        cons_line = ''
        line_line = ''
        future_top_down = defaultdict(list)
        end_bars = []
        span_tale = SpanTale()
        for pid, cid_pos_pairs in sorted(next_top_down.items(), key = lambda x: min(p for _, p in x[1])):
            num_children = len(cid_pos_pairs)
            if num_children < len(top_down[pid].children) or \
               num_children == 1 and span_tale.overlapped(cid_pos_pairs[0][1]) or \
               span_tale.overlapped(min(p for _, p in cid_pos_pairs), max(p for _, p in cid_pos_pairs)):
                future_top_down[pid].extend(cid_pos_pairs) # exclude ones with intersections
                continue
            if  num_children == 1:
                _, mid_pos = cid_pos_pairs[0]
                unit = '│'
                line_line += ((mid_pos - len(line_line)) * ' ') + unit
                if mid_pos in start_bars:
                    start_bars.remove(mid_pos)
                span_tale.add(mid_pos)
            else:
                mid_pos = 0
                cid_pos_pairs.sort(key = lambda x: x[1]) # left to right
                _, last_pos = cid_pos_pairs[0]
                start_pos = last_pos
                for cnt, (_, pos) in enumerate(cid_pos_pairs):
                    if pos in start_bars:
                        start_bars.remove(pos)
                    if cnt == 0:
                        unit = LC
                    elif cnt == num_children - 1:
                        unit += (pos - last_pos - 1) * '─' + RC
                    else:
                        unit += (pos - last_pos - 1) * '─' + MC
                    mid_pos += pos
                    last_pos = pos
                span_tale.add(start_pos, last_pos)
                mid_pos //= num_children
                offset = 1
                mid_mid = mid_pos
                # import pdb; pdb.set_trace()
                while any(-3 < mid_pos - bar < 3 for bar in start_bars): # avoid existing
                    mid_pos = mid_mid + offset
                    if offset > 0:
                        offset = 0 - offset
                    else:
                        offset = 1 - offset
                unit_half = mid_pos - start_pos
                if unit[unit_half] == MC:
                    unit = unit[:unit_half] + '┼' + unit[unit_half + 1:]
                else:
                    unit = unit[:unit_half] + MP + unit[unit_half + 1:]
                line_line += ((mid_pos - len(line_line)) * ' ')[:-unit_half] + unit
                # print('Comp:', unit)
            label = top_down[pid].label
            if pid == root_id:
                label += root_stamp
            cons_half = round(len(label) / 2)
            if cons_half:
                cons_line += ((mid_pos - len(cons_line)) * ' ')[:-cons_half] + label
            else:
                cons_line += ((mid_pos - len(cons_line)) * ' ') + label
            if pid in bottom_up:
                future_top_down[bottom_up[pid]].append((pid, mid_pos))
                end_bars.append(mid_pos)
        
        len_line = len(line_line)
        len_cons = len(cons_line)
        # prev_line = line_line
        # prev_cons = cons_line
        if len_line < pic_width:
            line_line += (pic_width - len_line) * ' '
        if len_cons < pic_width:
            cons_line += (pic_width - len_cons) * ' '
        if start_bars:
            new_line = ''
            new_cons = ''
            last_pos = 0
            for pos in sorted(start_bars):
                new_line += line_line[last_pos:pos] + '│'
                # if pos >= len(cons_line):
                #     print(prev_line)
                #     print(prev_cons)
                #     exit()
                #     cons_line += '<<<<<< overflow bar'
                #     continue
                if cons_line[pos] == ' ':
                    new_cons += cons_line[last_pos:pos] + '│'
                else:
                    new_cons += cons_line[last_pos:pos + 1]
                last_pos = pos + 1
            line_line = new_line + line_line[last_pos:]
            cons_line = new_cons + cons_line[last_pos:]
        start_bars.update(end_bars)

        str_lines.append(line_line)
        str_lines.append(cons_line)
        # print(line_line)
        # print(cons_line)
        next_top_down = future_top_down

    if reverse:
        str_lines.reverse()
    return str_lines