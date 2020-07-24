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

def _read_graph(graph):
    top_down = {}
    single_attachment = set()
    for nt in graph[1]:
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

    bottom = []
    for t in graph[0]:
        bottom.append((t.get('id'), t.get('word'), t.get('pos')))

    # CHECKED: p_node is the root_id
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

def validate_and_maintain(bottom_info, top_down, root_id, remove_cids, trace_dst):
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
    for cid in remove_cids + remove_bids:
        if cid in remove_bids:
            bid = cid - sum(td.bid < cid for td in trace_dst)
            assert bottom_info.pop(bid)[2] == '-NONE-'
            to_be_bids.remove(cid)
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

def _read_dpenn(tree):
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
    remove_cids = []
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
        if s_pid != nid and not top_down[s_pid].children: # empty node
            remove_cids.append(s_pid)

    validate_and_maintain(bottom, top_down, nid, remove_cids, trace_dst)
    vroot = top_down.pop(nid)
    assert vroot.label == 'VROOT'
    return bottom, top_down, get_sole_key(vroot.children)

def gap_degree(bottom, top_down, nid, bottom_is_bid = True):
    finally_return = True
    if isinstance(bottom, dict):
        if nid in bottom:
            return 0, set({bottom[nid]})
        finally_return = False
    elif bottom_is_bid:
        bottom = {bid: eid for eid, bid in enumerate(bottom)}
    else:
        bottom = {bid: eid for eid, (bid, _, _) in enumerate(bottom)}

    max_gap_num = 0
    coverage = set()
    for cid in top_down[nid].children:
        gap_num, child_coverage = gap_degree(bottom, top_down, cid)
        max_gap_num = max(gap_num, max_gap_num)
        coverage |= child_coverage

    if finally_return:
        return max_gap_num

    gap_num = 0
    last_nid = min(coverage) - 1
    for nid in sorted(coverage):
        if nid - last_nid > 1: # discontinuous
            gap_num += 1
        last_nid = nid
    max_gap_num = max(gap_num, max_gap_num)
    return max_gap_num, coverage

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

def _layer_base(bottom, top_down, completed_nodes, some_or_all, cnf_right, sub_suffix):
    bottom_up = {}
    right_hash = {}
    joint_rset = []
    bottom_flag = [True for _ in bottom]
    for p_node, td in top_down.items():
        if p_node in completed_nodes or not some_or_all(node in bottom for node in td.children):
            continue
        
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location_children.append((nid, node))

        num_children = len(location_children)
        for cid, (nid, node) in enumerate(location_children):
            if cid == 0:
                right = True
            else:
                right = False if cid == num_children - 1 else cnf_right
                if last_nid == nid - 1: # locality
                    joint_rset.append(nid)
                    assert last_right or not right # coherent
                    if last_right and not right: # agreeable_orients
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
    return right_hash, joint_rset, bottom_up

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
                if nid in joint_rset:
                    assert new_bottom.pop() == last_node
                    new_bottom.append(bottom_up.pop(node))
                else: # active swap
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
            elif not last_direc and not right: # passive swap
                if swap_distance != 1:
                    right_layer[-1] = True
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                elif swap_right_priority:
                    prev_node = new_bottom.pop(-2)
                    new_bottom += [node, prev_node]
                    swap_distance = 0
                else:
                    new_bottom.append(node)
            else:
                new_bottom.append(node)
            directional = True
        else:
            if last_direc:
                right = cnf_right
                if last_right and not cnf_right: # passive swap
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else:
                    new_bottom.append(node)
            else:
                right = last_right
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


def cross_signals(bottom_info, top_down, cnf_right,
                  aggressive = True,
                  swap_right_priority = None,
                  sub_prefix = '_',
                  pos_prefix = '#'):
    word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
    if swap_right_priority is None:
        swap_right_priority = not cnf_right

    some_or_all = has_multiple if aggressive else all
    bottom_tag = [node2tag[t] for t in bottom]
    sub_suffix = '.'

    layers_of_right = []
    layers_of_joint = []
    layers_of_label = []
    layers_of_direc = []
    completed_nodes = set()
    while len(bottom) > 1:
        (right_hash, joint_rset,
         bottom_up) = _layer_base(bottom,
                                  top_down,
                                  completed_nodes,
                                  some_or_all,
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

    if top_down:
        layers_of_right.append([cnf_right])
        layers_of_direc.append([False])
        layers_of_label.append([top_down[bottom.pop()].label])
    return word, bottom_tag, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc

def read_tiger_graph(graph, cnf_right):
    bottom_info, top_down, _ = _read_graph(graph[0])
    return cross_signals(bottom_info, top_down, cnf_right)

def read_disco_penn(tree, cnf_right):
    bottom_info, top_down, _ = _read_dpenn(tree)
    bottom_info = [(f'n_{bid}', w, t) for bid, w, t in bottom_info]
    new_top_down = {}
    for nid, td in top_down.items():
        children = {}
        for cid, ftag in td.children.items():
            children[f'n_{cid}'] = ftag
        new_top_down[f'n_{nid}'] = TopDown(td.label, children)
    return cross_signals(bottom_info, new_top_down, cnf_right)

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

def disco_tree(word, bottom_tag, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc):
    track_nodes = []
    terminals = []
    non_terminals = {}
    top_down = defaultdict(set)
    NTS = 500
    for tid, wd_tg in enumerate(zip(word, bottom_tag)):
        terminals.append(wd_tg)
        if layers_of_label[0][tid][0] not in '#_':
            bottom_unary = layers_of_label[0][tid].split('+')
            last_node = tid
            while bottom_unary:
                non_terminals[NTS] = bottom_unary.pop()
                top_down[NTS] = set({last_node})
                last_node = NTS
                NTS += 1
            track_nodes.append(NTS - 1)
        else:
            track_nodes.append(tid)
            
    non_terminal_start = NTS
    def _combine(child_node, parent_node, non_terminals, top_down):
        if child_node < NTS or non_terminals[child_node][0] not in '#_':
            top_down[parent_node].add(child_node)
            safe_label = None
        else:
            top_down[parent_node] |= top_down.pop(child_node)
            safe_label = non_terminals.pop(child_node)[1:]
            safe_label = non_terminals[parent_node].endswith(safe_label)
        return safe_label

    for lid, (lr, lj, ld) in enumerate(zip(layers_of_right, layers_of_joint, layers_of_direc)):
        for nid, (right, direc) in enumerate(zip(lr, ld)):
            if nid == 0:
                offset = 1
            elif lj[nid - 1]: # joint
                if last_right and not right:
                    lhs_nid = nid - offset
                    lhs_node = track_nodes.pop(lhs_nid)
                    rhs_node = track_nodes.pop(lhs_nid)
                    labels = layers_of_label[lid + 1][lhs_nid]
                    labels = [labels] if labels[0] in '#_' else labels.split('+')
                    non_terminals[non_terminal_start] = labels.pop()
                    _combine(lhs_node, non_terminal_start, non_terminals, top_down) # TODO: safe_label validate
                    _combine(rhs_node, non_terminal_start, non_terminals, top_down)
                    while labels:
                        non_terminal_start += 1
                        non_terminals[non_terminal_start] = labels.pop()
                        top_down[non_terminal_start] = set({non_terminal_start - 1})
                    track_nodes.insert(lhs_nid, non_terminal_start)
                    non_terminal_start += 1
                    offset += 1
            elif last_direc and last_right and (not direc or direc and not right) or not last_direc and direc and not right: # cross
                lhs_nid = nid - offset
                rhs_nid = lhs_nid + 1
                track_nodes[lhs_nid], track_nodes[rhs_nid] = track_nodes[rhs_nid], track_nodes[lhs_nid]

            last_right = right
            last_direc = direc

    return top_down, terminals, non_terminals

from data.delta import get_logits as _get_logits
def xlogit_gen(label_layer, right_layer, direc_layer, joint_layer, current_joints, next_joints):
    for nid, (label, right, direc) in enumerate(zip(label_layer, right_layer, direc_layer)):
        is_phrase = label[0] not in '#_'
        is_joint = current_joints and current_joints[nid]
        if nid and (last_right or not right):
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

from data.trapezoid import trapezoid_to_layers
from numpy import asarray
from data.delta import get_rgt, get_jnt, get_dir
def unzip_xlogit(cindex, xtypes):
    xtypes = asarray(xtypes)
    rights = get_rgt(xtypes)
    joints = get_jnt(xtypes)
    direcs = get_dir(xtypes)
    joints = trapezoid_to_layers(joints, cindex, cindex, big_endian = False)
    layers_of_direc = trapezoid_to_layers(direcs, cindex, cindex, big_endian = False)
    layers_of_right = trapezoid_to_layers(rights, cindex, cindex, big_endian = False)
    layers_of_joint = []
    for lower_rights, upper_joints in zip(layers_of_right, joints[1:]):
        joint_layer = []
        upper_joints = list(upper_joints)
        upper_joints.reverse() # pop() should be like dequeue
        for rid, right in enumerate(lower_rights):
            if rid:
                if not last_right and right:
                    joint_layer.append(False)
                else:
                    joint = upper_joints.pop()
                    joint_layer.append(joint)
                    if last_right and not right and not joint: # a swap
                        assert not upper_joints.pop()
            last_right = right
        if upper_joints:
            assert not upper_joints.pop()
        layers_of_joint.append(joint_layer)
              
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