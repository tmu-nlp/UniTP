need_none = {
    "`` Nasty innuendoes , '' says *-2 John Siegal , Mr. Dinkins 's issues director , `` designed * *-1 to prosecute a case of political corruption that *T*-74 simply does n't exist . ''": ((1, 0, 0), 1, ()),
    "`` A very striking illusion , '' Mr. Hyman says *-1 now , his voice dripping with skepticism , `` but an illusion nevertheless . ''": ((4, 0, 1), 1, ()),
}

remove_none = {"They 're going *-1 to decide what their employees can *RNR*-2 or can not *RNR*-2 read *T*-3 . ''": (1,1,1,1,1,1,1,1,2,2)}

need_prn = {
    "But , says 0 *T*-1 chief investigator Tom Smith , this `` does not translate into support for conservatism in general or into conservative positions on feminist and civil rights issues . ''": ((1,), 3),
    "And , `` more importantly , '' he says *T*-1 , `` the debt burden measured * other ways is not really in uncharted waters . ''": ((4,), 5),
    "Mr. Noriega might have fallen of his own weight in 1988 because of Panama 's dire economic situation , says 0 *T*-1 Mr. Moss , but increasing external pressure has only given him additional excuses for repression , and a scapegoat for his own mismanagement .": ((1,), 3),
    "`` North , '' the document went on *T*-2 , *-1 referring to Oliver North , `` has told Noriega 's representative that U.S. law forbade such actions .": ((2,), 5),
}

shift_trace = {
    "THE MILEAGE RATE allowed * for business use of a car in 1989 has risen to 25.5 cents a mile for the first 15,000 from 24 cents in 1988 , the IRS says 0 *T*-1 ; the rate stays 11 cents for each added mile .": ((0,), (0, 0)),
    "`` Wage increases and overall compensation increases are beginning *-1 to curl upward a little bit , '' said *T*-2 Audrey Freedman , a labor economist at the Conference Board , a business research organization .": ((), (1,)),
    "`` ` God has not yet ordained that I should have earnings , ' he tells his worried mother *T*-1 . ''": ((), (2,)),
    "`` We 're trying *-1 to take the imagination and talent of our engineers and come up with new processes for industry , '' says *T*-2 Vincent Salvatori , QuesTech 's chief executive .": ((), (1,)),
    "`` Our ordnance business has been hurt *-1 very badly by the slowdown , '' says *T*-2 Arch Scurlock , TransTechnology 's chairman .": ((), (1,)),
    "When a court decides that a particular actor 's conduct was culpable and so extends the definition of insider trading * to reach this conduct *T*-3 , it does not see the potentially enormous number of other cases that *T*-2 will be covered *-1 by the expanded rule .": ((0,), (0, 0)),
}

add_trace = {
    "Both Mr. Brown , the state 's most influential legislator , and Gov. Deukmejian favor a temporary sales tax increase -- should more money *ICH*-2 be needed *-1 than the state can raise *?* from existing sources and the federal government .": ((1, 3, 1, 1, 1, 2), '-2'),
    "Now , with the crowd in the analysis room smelling figurative blood , the only question seemed *-1 to be how fast Mr. Kasparov could win game two *T*-2 .": ((5, 1, 1, 1, 1, 0), '-2'),
    "In the corporate realm , the 1986 law abolished the investment-tax credit , scaled back use of an accounting method that *T*-1 allowed large contractors to defer taxes until a project was completed *-1 and strengthened the so-called alternative minimum tax , a levy 0 *T*-2 to ensure 0 all money-making businesses pay some federal tax .": ((3, 2, 2, 1, 1, 1, 0), '0'),
    "J.P. Bolduc , vice chairman of W.R. Grace & Co. , which *T*-10 holds a 83.4 % interest in this energy-services company , was elected *-10 a director .": ((0, 2, 1, 1, 2, 0), '1')
}

remove_trace = {"As Mr. Colton of the NAHB acknowledges *T*-1 : `` Government is not going *-2 to solve the problem ... .": ()}

delete_node = {
    "But as Drexel analyst Linda Dunn *T*-1 notes , its properties will be developed *-2 over 15 to 20 years .":((1,0,1,1), (1,0,1,1))
}

add_s_and_shift_trace = {"This calamity is `` far from over , '' he says 0 *T*-1 .": ((0,), 2, (), (0,))}

from nltk.tree import Tree

def wrap_with_label_fn(tree, path, num, label):
    last = path[-1]
    path = path[:-1]
    children = []
    for _ in range(num):
        children.append(tree[path].pop(last))
    tree[path].insert(last, Tree(label, children))

def shift_fn(tree, s_path, d_path):
    s_label = tree[s_path].label()
    s_label, tid = s_label.rsplit('-', 1)
    tree[s_path].set_label(s_label)
    tree[d_path].set_label(tree[d_path].label() + '-' + tid)

_CMD_TAG = 0
_CMD_BOL = 1
_CMD_EOL = 2
E_DISCO = '*T*', '*ICH*', '*EXP*', '*RNR*'

def remove_irrelevant_trace(tree):
    bottom = list(enumerate(tree.pos()))
    bottom.reverse()
    for bid, (word, tag) in bottom:
        if tag != '-NONE-' or any(word.startswith(tc) for tc in E_DISCO):
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

from data.delta import remove_eq, XRB2brackets
def fix_for_dptb(tree):
    sent = ' '.join(tree.leaves()).strip()
    if sent in need_none:
        d_path, loc, s_path = need_none[sent]
        segs = tree[s_path].label().split('-')
        label = segs.pop(0); tid = segs.pop()
        tree[d_path].insert(loc, Tree.fromstring(f'({label} (-NONE- *T*-{tid}))'))
    elif sent in shift_trace:
        s_path, d_path = shift_trace[sent]
        shift_fn(tree, s_path, d_path)
    elif sent in need_prn:
        if sent.startswith('Mr.'):
            wrap_with_label_fn(tree, (2,), 2, 'SINV')
        path, num = need_prn[sent]
        wrap_with_label_fn(tree, path, num, 'PRN')
    elif sent in add_trace:
        path, tid = add_trace[sent]
        tree[path].set_label(tree[path].label() + tid)
        if tid.isdigit():
            tree[path[:-1] + (1, 0, 0, 0)] += tid
    elif sent in remove_trace:
        path = remove_trace[sent]
        label, _ = tree[path].label().rsplit('-', 1)
        tree[path].set_label(label)
    elif sent in add_s_and_shift_trace:
        path, num, s_path, d_path = add_s_and_shift_trace[sent]
        wrap_with_label_fn(tree, path, num, 'S')
        shift_fn(tree, s_path, d_path)
    elif sent in delete_node:
        for path in delete_node[sent]:
            tree[path].extend(tree[path].pop())
    elif sent in remove_none:
        del tree[remove_none[sent]]
    remove_irrelevant_trace(tree)
    return dict(w_fn = XRB2brackets, nt_fn = remove_eq)

from data.cross import TopDown, C_VROOT, do_nothing, descendant_path, find_labeled_on_path, boundary
def _preorder(tree, w_fn = do_nothing, t_fn = do_nothing, nt_fn = do_nothing):
    if tree.height() < 3:
        assert len(tree) == 1
        yield _CMD_TAG
        yield w_fn(tree[0]), t_fn(tree.label())
    else:
        for child in tree:
            yield from _preorder(child, w_fn, t_fn, nt_fn)
        yield _CMD_BOL
        for child in reversed(tree):
            yield nt_fn(child.label())
        yield _CMD_EOL
        yield nt_fn(tree.label())

def direct_read(tree, *, cid = 1, nid = 0):
    this_nid = nid
    bt = {}
    td = {}
    if tree.height() > 2:
        children = {}
        for subtree in tree:
            nid -= 1
            children[nid if subtree.height() > 2 else cid] = None
            _bt, _td, cid, nid = direct_read(subtree, cid = cid, nid = nid)
            bt.update(_bt)
            td.update(_td)
        td[this_nid] = TopDown(tree.label(), children)
    else:
        bt[cid] = tree[0], tree.label()
    if this_nid == 0:
        bt = sorted((cid, wd, tg) for cid, (wd, tg) in bt.items())
        return bt, td
    return bt, td, cid + 1, nid

from collections import namedtuple, defaultdict
TraceRef = namedtuple('TraceRef', 'pid, cid, lhs, rhs')
TraceID = namedtuple('TraceID', 'typ, tid, pid, cid, bid')

def order_trace_ids_gen(top_down, trace_refs, trace_ids):
    active_trace_ids = []
    trace_dependency = {} # cascade trace chain
    for tid, tds in trace_ids.items(): # index identity
        if tid in trace_refs:
            active_trace_ids.append(tid)
            for sid, ts in trace_refs.items(): # reference
                if tid == sid: continue
                for td in tds:
                    if ts.cid == td.pid or descendant_path(top_down, ts.cid, td.pid):
                        trace_dependency[tid] = sid # [id] = ref

    priority = defaultdict(int)
    for tid in active_trace_ids:
        anti_loop = tid
        while tid in trace_dependency:
            tid = trace_dependency[tid]
            priority[tid] += 1
            if tid == anti_loop: break
    active_trace_ids.sort(key = lambda x: priority[x])

    # select the nearest for multi-attachment
    for tid in active_trace_ids:
        tr = trace_refs.pop(tid)
        tds = trace_ids.pop(tid)
        if len(tds) > 1:
            distances = {}
            lhs, rhs = tr.lhs, tr.rhs
            for ti, td in enumerate(tds):
                d_bid = td.bid; assert tid == td.tid
                if d_bid < lhs: dist = lhs - d_bid, 0
                elif rhs < d_bid: dist = d_bid - rhs, 0
                else: dist = 0, min(d_bid - lhs, rhs - d_bid)
                distances[ti] = dist
            ti = min(distances, key = distances.get)
            trace_ids[tid] = [t for i,t in enumerate(tds) if ti != i]
        else:
            ti = 0
        yield tds[ti] + tr
        # No need for DPTB:
        # if tid in trace_dependency and (sid := trace_dependency.pop(tid)) in trace_refs:
        #     n_pid, n_cid, n_lhs, n_rhs = trace_refs.pop(sid)
        #     if tr.lhs < n_lhs: n_lhs = tr.lhs
        #     if n_rhs < tr.rhs: n_rhs = tr.rhs
        #     trace_refs[sid] = TraceRef(n_pid, n_cid, n_lhs, n_rhs)

def check_trace_id(valid_trace_types, wd, tg, trace_ids, nid):
    if tg == '-NONE-':
        trace_type, tid = wd.split('-')
        if trace_type in valid_trace_types and tid.isdigit():
            trace_ids[nid] = trace_type, tid

def split_label_ftag_trace(item, trace_refs, ftags, nid):
    if '-' in item:
        segments = item.split('-')
        item = segments.pop(0)
        if segments[-1].isdigit(): # for both '-='
            trace_refs[nid] = segments.pop()
        if segments:
            ftags[nid] = '-'.join(segments)
    return item

def read_materials(tree, *catch_nodes, **wtnt_fns):
    bottom = {}
    top_down = {}
    ftags = {}
    trace_refs = {} # reference
    trace_ids = defaultdict(list) # trace identity
    catch_nodes = {k: set() for k in catch_nodes}
    short_memory = defaultdict(set)
    nt_shifter = len(tree.leaves())
    nt_start = nt_shifter + 1
    tree = Tree(C_VROOT, [tree])
    for item in _preorder(tree, **wtnt_fns):
        if isinstance(item, int):
            status = item
            if status == _CMD_BOL:
                nid = nt_start + len(top_down)
                top_down[nid] = []
                short_memory[None].add(nid)
        elif status == _CMD_TAG: # item is a (word,  tag) tuple
            nid = len(bottom) + 1 # start from 1, 0 for ROOT
            bottom[nid] = (wd, tg) = item
            short_memory[tg].add(nid)
            check_trace_id(E_DISCO, wd, tg, trace_ids, nid)
        elif status == _CMD_BOL: # item is a tag or a label
            cnid = max(short_memory[item]) # cnid never belongs to item
            short_memory[item].remove(cnid)
            if not short_memory[item]:
                short_memory.pop(item)
            top_down[nid].append(cnid) # cnid belongs to nid
        elif status == _CMD_EOL:
            # item is the parent label
            short_memory[item] |= short_memory.pop(None)
            item = split_label_ftag_trace(item, trace_refs, ftags, nid)
            if item in catch_nodes:
                catch_nodes[item].add(nid)

            children = {}
            for cnid in top_down[nid]:
                children[cnid] = ftags.pop(cnid, None)

                if cnid in trace_refs: # register formal reference
                    tid = trace_refs.pop(cnid)
                    # if tid not in trace_refs or not top_down[trace_refs[tid].cid].label.startswith('WH'): # wh-movement has the priority
                        # if tid in trace_refs:
                        #     breakpoint()
                        #     print(tid)
                    trace_refs[tid] = TraceRef(nid, cnid, *boundary(top_down, cnid))

                if cnid in trace_ids: # register formal identity
                    if len(ty_id := trace_ids.pop(cnid)) == 2: # for raw, change cnid to nid
                        trace_ids[nid] = ty_id + (cnid,)
                    elif len(ty_id) == 3:
                        typ, tid, bid = ty_id # to be mature, nid into tid
                        trace_ids[tid].append(TraceID(typ, tid, nid, cnid, bid))

            top_down[nid] = TopDown(item, children)
    assert not ftags or nid in ftags
    assert len(short_memory) == 1
    assert nid in short_memory[C_VROOT]
    get_inode = lambda x: x if x < nt_start else (nt_shifter - x)
    return bottom, top_down, nid, get_inode, nt_start, trace_refs, trace_ids, catch_nodes

from utils.param_ops import get_sole_key
def read_tree(tree, *,
              adjust_fn = fix_for_dptb,
              return_type_count = False):
    wtnt_fns = {}
    if callable(adjust_fn):
        wtnt_fns.update(adjust_fn(tree))

    bottom, top_down, nid, get_inode, nt_start, trace_refs, trace_ids, catch_nodes = read_materials(tree, 'PRN', **wtnt_fns)
    remaining_PRN_nodes = catch_nodes.pop('PRN')

    # cross trace along the bottom (ordered and reversed for bottom.pop(i) stability)
    history = {}
    type_count = defaultdict(int)
    for typ, tid, d_pid, d_cid, d_bid, s_pid, s_cid, lhs, rhs in order_trace_ids_gen(top_down, trace_refs, trace_ids):
        d_pid = history.pop(d_cid, d_pid)
        s_ftag = top_down[s_pid].children.pop(s_cid)
        d_ftag = top_down[d_pid].children.pop(d_cid)
        v_wd, v_tg = bottom.pop(d_bid)
        assert v_wd.endswith(tid)
        assert v_tg == '-NONE-'
        if s_ftag and d_ftag:
            ftag = s_ftag if s_ftag == d_ftag else (s_ftag + ':' + d_ftag)
        else:
            ftag = s_ftag or d_ftag
        top_down[d_pid].children[s_cid] = ftag
        history[s_cid] = d_pid # add s_cid as a d_pid child
        if lhs <= d_bid <= rhs and (loc := find_labeled_on_path(top_down, s_cid, 'PRN', d_pid)): # PRN
            s_cid, s_ccid = loc
            ftag = top_down[s_cid].children.pop(s_ccid)
            top_down[s_pid].children[s_ccid] = ftag
            remaining_PRN_nodes.remove(s_ccid)
            typ += '-PRN'
        type_count[typ] += 1

    validate_and_maintain(bottom, top_down, nid, nt_start)
    vroot = top_down.pop(nid)
    assert vroot.label == C_VROOT
    root_id = get_sole_key(vroot.children)

    condense = {bid: eid for eid, bid in enumerate(sorted(bottom), 1) if eid != bid}
    bottom = [(condense.get(bid, bid), w, t) for bid, (w, t) in bottom.items()]
    new_top_down = {}
    for nid, td in top_down.items():
        children = {}
        for cid, ftag in td.children.items():
            children[get_inode(condense.get(cid, cid))] = ftag
        nid = 0 if nid == root_id else get_inode(nid)
        new_top_down[nid] = TopDown(td.label, children)
    top_down = new_top_down

    if return_type_count:
        return bottom, top_down, type_count, trace_refs, trace_ids
    return bottom, top_down

def read_graph(tree, *,
               adjust_fn = fix_for_dptb,
               return_type_count = False):
    wtnt_fns = {}
    if callable(adjust_fn):
        wtnt_fns.update(adjust_fn(tree))

    bottom, top_down, nid, get_inode, nt_start, trace_refs, trace_ids, catch_nodes = read_materials(tree, 'PRN', **wtnt_fns)
    remaining_PRN_nodes = catch_nodes.pop('PRN')

def validate_and_maintain(bottom_info, top_down, root_id, nt_start):
    cids = set()
    nids = [root_id]
    bottom_up = {}
    should_be_bids = set()
    existing_bids = set(bottom_info)
    seen_nids = set()
    while nids:
        for nid in nids:
            if nid < nt_start:
                should_be_bids.add(nid)
            elif nid not in top_down:
                raise ValueError(f'Not found nid({nid}) in top_down[\'{set(top_down)}\']')
            seen_nids.add(nid)
            for cid in top_down[nid].children:
                if cid < nt_start:
                    should_be_bids.add(cid)
                else:
                    cids.add(cid)
                bottom_up[cid] = nid
        nids = cids
        cids = set()
    remove_bids = [bid for bid, (_, tag) in bottom_info.items() if tag == '-NONE-']
    remove_cids = [s_pid for s_pid in top_down if s_pid != root_id and not top_down[s_pid].children]
    for cid in remove_cids + remove_bids:
        if cid in remove_bids:
            assert bottom_info.pop(cid) #[1] == '-NONE-'
            should_be_bids.remove(cid)
            existing_bids.remove(cid)
        else:
            top_down.pop(cid) # maintain
            seen_nids.remove(cid)
        pid = bottom_up.pop(cid)
        top_down[pid].children.pop(cid)
        while not top_down[pid].children: # empty again
            top_down.pop(pid)
            cid = pid
            seen_nids.remove(cid)
            pid = bottom_up.pop(cid)
            top_down[pid].children.pop(cid) # maintain
            
    if diff_bids := should_be_bids ^ existing_bids:
        if lacking_bids := should_be_bids - existing_bids:
            msg = f'Lacking bids: {lacking_bids}'
        else:
            msg = f'Redundant bids: {diff_bids - lacking_bids}'
        raise ValueError(msg)
    elif redundant_nids := top_down.keys() - seen_nids:
        for nid in redundant_nids:
            _, children = top_down.pop(nid) # maintain
            safe = True
            for cid in children:
                if cid < nt_start:
                    safe &= cid not in existing_bids
                else:
                    safe &= cid in redundant_nids
                if not safe:
                    break
            if not safe:
                raise ValueError(f'Redundant nids: {redundant_nids}')

# E_PENN_PUNCT = ',', '.', '``', "''", ':', '-RRB-', '-LRB-'
# def sort_leftover(lhs, rhs, leftover_gen):
#     def dist(x):
#         if x < lhs:
#             return lhs - x
#         elif x > rhs:
#             return x - rhs
#         return 0
#     leftover = [(x, dist(x)) for x in leftover_gen]
#     leftover.sort(key = lambda x: x[1])
#     leftover_x = []
#     for x, y in leftover:
#         if y == 0:
#             leftover_x.append(x)
#         elif x == lhs - 1:
#             lhs = x
#             leftover_x.append(x)
#         elif x == rhs + 1:
#             rhs = x
#             leftover_x.append(x)
#         else:
#             break
#     return leftover_x
#     # for better continuity
#     if adjust_punct:
#         leftover_x = (x for x in top_down[s_pid].children.keys() - set({d_pid}) if x < nt_start)
#         leftover_x = sort_leftover(lhs, rhs, leftover_x)
#         if leftover_x and all(bottom[x][2] in E_PENN_PUNCT for x in leftover_x):
#             for x in leftover_x:
#                 top_down[d_pid].children[x] = top_down[s_pid].children.pop(x)
# nids = [nid]
# cids = []
# coverage = []
# while nids:
#     for nid in nids:
#         for cid in top_down[nid].children:
#             if cid in top_down:
#                 cids.append(cid)
#             else:
#                 assert cid < nt_start
#                 coverage.append(cid)
#     nids = cids
#     cids = []
# return min(coverage), max(coverage)