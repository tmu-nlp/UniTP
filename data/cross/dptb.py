change_trace = {
    "`` Nasty innuendoes , '' says *-2 John Siegal , Mr. Dinkins 's issues director , `` designed * *-1 to prosecute a case of political corruption that *T*-74 simply does n't exist . ''": ((1, 0, 0, 1, 0, 0), '*T*'),
    "`` A very striking illusion , '' Mr. Hyman says *-1 now , his voice dripping with skepticism , `` but an illusion nevertheless . ''": ((4, 0, 1, 1, 0, 0), '*T*'),
}

remove_none = {"They 're going *-1 to decide what their employees can *RNR*-2 or can not *RNR*-2 read *T*-3 . ''": (1,1,1,1,1,1,1,1,2,2),
    "Mrs. Yeargin says 0 she pleaded guilty because she realized 0 it *EXP*-2 would no longer be possible *-1 to win reinstatement , and because she was afraid of further charges .": (1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 0),
    "IT *EXP*-2 PAYS *-1 to follow a management career path -- even at law firms .": (1, 1, 0)}

add_node = {
    "But , says 0 *T*-1 chief investigator Tom Smith , this `` does not translate into support for conservatism in general or into conservative positions on feminist and civil rights issues . ''": (((1,), 3, 'PRN'),),
    "And , `` more importantly , '' he says *T*-1 , `` the debt burden measured * other ways is not really in uncharted waters . ''": (((4,), 5, 'PRN'),),
    "Mr. Noriega might have fallen of his own weight in 1988 because of Panama 's dire economic situation , says 0 *T*-1 Mr. Moss , but increasing external pressure has only given him additional excuses for repression , and a scapegoat for his own mismanagement .": (((2,), 2, 'SINV'), ((1,), 3, 'PRN'),),
    "`` North , '' the document went on *T*-2 , *-1 referring to Oliver North , `` has told Noriega 's representative that U.S. law forbade such actions .": (((2,), 5, 'PRN'),),
    "Mr. Kasparov 's continuation was slower but , in the end , just as deadly .": (((1,), 3, 'VP'),),
    "It 's a world leader in semiconductors , but behind the U.S. in * making the computers that *T*-1 use those chips .": (((1, 3), 2, 'VP'),),
    "And only last week the newspaper Vanguardia Liberal in the city of Bucaramanga was bombed *-1 , and its installations destroyed *-2 .": (((2,), 2, 'S'),),
    "Import growth from the year-earlier months slowed to 16 % in July and 7.1 % in August , compared with an average growth rate of 26 % in the first half .": (((1, 1), 3, 'PP'),),
    "In the Grand Boulevard Plaza developed * by Matanky Realty Group in Chicago 's Third Ward , opposite the Robert Taylor Homes , 29 % of the stores to date have been leased *-1 to blacks and 14 % to members of other minority groups .": (((1,), 3, 'S'), ((1,), 3, 'S')),
}

adjust_exp = {
    "`` It *EXP*-2 's one thing * to say 0 you can sterilize , and another * to then successfully pollinate the plant , '' he said *T*-1 .": (lambda x: wrap_with_label_fn(x, (1, 0), 1, 'S'), lambda x: shift_node_fn(x, (1, 1, 0), (1, 0, 1)), lambda x: del_node_fn(x, (1, 1)), lambda x: rename_node_fn(x, (1, 3), 'S')),
    "It *EXP*-1 will take several weeks * to repair the bridge , and several months * to repair some of the 101 connections .": (lambda x: wrap_with_label_fn(x, (0,), 2, 'S'), lambda x: wrap_with_label_fn(x, (0, 1, 1, 1), 3, 'S'), lambda x: shift_node_fn(x, (0, 1, 1, 1), (1, )), lambda x: del_node_fn(x, (1,)), lambda x: rename_node_fn(x, (3,), 'S')),
    "It *EXP*-1 took me a half-hour * to move 10 feet from my parking spot in an outer lot to an aisle , and an additional hour * to reach an inner roadway a half-block away .": (lambda x: del_node_fn(x, (1,)), lambda x: wrap_with_label_fn(x, (0,), 2, 'S'), lambda x: rename_node_fn(x, (3,), 'S')),
}

shift_trace = {
    "THE MILEAGE RATE allowed * for business use of a car in 1989 has risen to 25.5 cents a mile for the first 15,000 from 24 cents in 1988 , the IRS says 0 *T*-1 ; the rate stays 11 cents for each added mile .": ((0,), (0, 0)),
    "`` Wage increases and overall compensation increases are beginning *-1 to curl upward a little bit , '' said *T*-2 Audrey Freedman , a labor economist at the Conference Board , a business research organization .": ((), (1,)),
    "`` ` God has not yet ordained that I should have earnings , ' he tells his worried mother *T*-1 . ''": ((), (2,)),
    "`` We 're trying *-1 to take the imagination and talent of our engineers and come up with new processes for industry , '' says *T*-2 Vincent Salvatori , QuesTech 's chief executive .": ((), (1,)),
    "`` Our ordnance business has been hurt *-1 very badly by the slowdown , '' says *T*-2 Arch Scurlock , TransTechnology 's chairman .": ((), (1,)),
    "When a court decides that a particular actor 's conduct was culpable and so extends the definition of insider trading * to reach this conduct *T*-3 , it does not see the potentially enormous number of other cases that *T*-2 will be covered *-1 by the expanded rule .": ((0,), (0, 0)),
    "Philip Morris , trying *-1 to revive the Benson & Hedges franchise , put the account up for review in 1986 .": ((0), (0, 0)),
    "Partners in the survey who *T*-1 devote most of their time to *-2 practicing law earned an average of about $ 217,000 *U* .": ((0), (0, 0))
}

add_trace = {
    "Both Mr. Brown , the state 's most influential legislator , and Gov. Deukmejian favor a temporary sales tax increase -- should more money *ICH*-2 be needed *-1 than the state can raise *?* from existing sources and the federal government .": ((1, 3, 1, 1, 1, 2), '-2'),
    "Now , with the crowd in the analysis room smelling figurative blood , the only question seemed *-1 to be how fast Mr. Kasparov could win game two *T*-2 .": ((5, 1, 1, 1, 1, 0), '-2'),
    "In the corporate realm , the 1986 law abolished the investment-tax credit , scaled back use of an accounting method that *T*-1 allowed large contractors to defer taxes until a project was completed *-1 and strengthened the so-called alternative minimum tax , a levy 0 *T*-2 to ensure 0 all money-making businesses pay some federal tax .": ((3, 2, 2, 1, 1, 1, 0), '0'),
    "J.P. Bolduc , vice chairman of W.R. Grace & Co. , which *T*-10 holds a 83.4 % interest in this energy-services company , was elected *-10 a director .": ((0, 2, 1, 1, 2, 0), '1'),
    "Merck 's profit climbed 25 % , Warner-Lambert 's 22 % and Eli Lilly 's 24 % .": ((4, 1), '=2')
}

remove_trace = {
    "As Mr. Colton of the NAHB acknowledges *T*-1 : `` Government is not going *-2 to solve the problem ... .": (),
    "Rival Boston Herald columnist Howie Carr , who *T*-1 usually rails at Statehouse `` hacks '' and nepotism , argued that the new drawings were designed *-2 *-3 to hide Mr. Madden 's `` rapidly growing forehead '' and the facial defects of `` chinless '' Dan Shaughnessy , a Globe sports columnist .": (1, 1, 1, 1, 1),}

delete_node = {
    "But as Drexel analyst Linda Dunn *T*-1 notes , its properties will be developed *-2 over 15 to 20 years .":((1,0,1,1,0), (1,0,1,1,1)),
    "The spy network would serve two clients *NOT* : the Panamanian government , by * monitoring political opponents in the region , and the U.S. , by * tracking the growing Communist influence in the unions organized * at United Fruit Co. 's banana plantations in Bocas del Toros and Puerto Armuelles .": ((1, 1, 2),)
}

add_s_and_shift_trace = {"This calamity is `` far from over , '' he says 0 *T*-1 .": ((0,), 2, (), (0,))}

short_circuits = {
    "Mr. Kasparov 's continuation was slower but , in the end , just as deadly .",
    "It 's a world leader in semiconductors , but behind the U.S. in * making the computers that *T*-1 use those chips .",
    "And only last week the newspaper Vanguardia Liberal in the city of Bucaramanga was bombed *-1 , and its installations destroyed *-2 .",
    "Import growth from the year-earlier months slowed to 16 % in July and 7.1 % in August , compared with an average growth rate of 26 % in the first half .",
    "In the Grand Boulevard Plaza developed * by Matanky Realty Group in Chicago 's Third Ward , opposite the Robert Taylor Homes , 29 % of the stores to date have been leased *-1 to blacks and 14 % to members of other minority groups .",
    "The spy network would serve two clients *NOT* : the Panamanian government , by * monitoring political opponents in the region , and the U.S. , by * tracking the growing Communist influence in the unions organized * at United Fruit Co. 's banana plantations in Bocas del Toros and Puerto Armuelles ."
}
short_circuits.update(adjust_exp.keys())

from nltk.tree import Tree

def wrap_with_label_fn(tree, path, num, label):
    last = path[-1]
    path = path[:-1]
    children = []
    for _ in range(num):
        children.append(tree[path].pop(last))
    tree[path].insert(last, Tree(label, children))

def shift_trace_fn(tree, s_path, d_path):
    s_label = tree[s_path].label()
    s_label, tid = s_label.rsplit('-', 1)
    tree[s_path].set_label(s_label)
    tree[d_path].set_label(tree[d_path].label() + '-' + tid)

def del_node_fn(tree, path):
    children = reversed(tree[path])
    last = path[-1]
    path = path[:-1]
    tree[path].pop(last)
    for child in children:
        tree[path].insert(last, child)

def shift_node_fn(tree, s_path, d_path):
    tree[d_path[:-1]].insert(d_path[-1], tree[s_path[:-1]].pop(s_path[-1]))

def rename_node_fn(tree, path, label):
    tree[path].set_label(label)

E_DISCO = '*T*', '*ICH*', '*EXP*', '*RNR*'
A_MOVEMENT = '*',
E_GRAPH = E_DISCO + A_MOVEMENT#  '*PPA*'

def typed_trace(types, word):
    if '-' in word:
        typ, trace_id = word.split('-')
        if typ in types and trace_id.isdigit():
            return typ, trace_id

def remove_irrelevant_trace(tree, traces):
    bottom = list(enumerate(tree.pos()))
    bottom.reverse()
    for bid, (word, tag) in bottom:
        if tag != '-NONE-' or typed_trace(traces, word):
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
def fix_for_ptb(tree, short_circuits):
    sent = ' '.join(tree.leaves()).strip()
    if sent in short_circuits: return
    if sent in change_trace:
        path, typ = change_trace[sent]
        _, tid = tree[path].split('-')
        tree[path] = typ + '-' + tid
    elif sent in shift_trace:
        s_path, d_path = shift_trace[sent]
        shift_trace_fn(tree, s_path, d_path)
    elif sent in add_node:
        for path, num, label in add_node[sent]:
            wrap_with_label_fn(tree, path, num, label)
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
        shift_trace_fn(tree, s_path, d_path)
    elif sent in delete_node:
        for path in delete_node[sent]:
            del_node_fn(tree, path)
    elif sent in remove_none:
        del tree[remove_none[sent]]
    elif sent in adjust_exp:
        for fn in adjust_exp[sent]:
            fn(tree)


def fix_for_dptb(tree):
    fix_for_ptb(tree, short_circuits)
    remove_irrelevant_trace(tree, E_DISCO)
    return dict(w_fn = XRB2brackets, nt_fn = remove_eq)

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

_CMD_TAG = 0
_CMD_BOL = 1
_CMD_EOL = 2
from data.cross import TopDown, C_VROOT, do_nothing, PathFinder, boundary, height_gen
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

from collections import namedtuple, defaultdict, Counter
TraceRef = namedtuple('TraceRef', 'pid, cid, lhs, rhs')
TraceID = namedtuple('TraceID', 'typ, tid, pid, cid, bid')
from utils.graph import directed_graph

def trace_priority(trace_refs, trace_ids):
    active_tid_priority = {}
    directed_edges = set() # cascade trace chain
    for tid, tds in trace_ids.items(): # index identity
        if tid in trace_refs:
            active_tid_priority[tid] = 0
            for td in tds:
                for rid, tr in trace_refs.items(): # reference
                    if tid != rid and rid in trace_ids and tr.lhs <= td.bid <= tr.rhs:
                        directed_edges.add((tid, rid)) # [id] = ref

    _, loops = directed_graph(active_tid_priority, directed_edges)

    return active_tid_priority, loops

def tree_trace_gen(trace_refs, trace_ids):
    priority, _ = trace_priority(trace_refs, trace_ids)
    # For tree, detach both tref and tid and save c2p_history[tref].
    # The depandant will find its relocated parent in c2p_history.
    # select the nearest for multi-attachment
    for tid in sorted(priority, key = priority.get, reverse = True):
        tr = trace_refs.pop(tid)
        tds = trace_ids.pop(tid)
        if len(tds) > 1:
            distances = {}
            lhs, rhs = tr.lhs, tr.rhs
            for ti, td in enumerate(tds):
                i_bid = td.bid; assert tid == td.tid
                if i_bid < lhs: dist = lhs - i_bid, 0
                elif rhs < i_bid: dist = i_bid - rhs, 0
                else: dist = 0, min(i_bid - lhs, rhs - i_bid)
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

def init_trace_id(valid_trace_types, wd, tg, trace_ids, nid):
    if tg == '-NONE-' and (typ_tid := typed_trace(valid_trace_types, wd)):
        trace_ids[nid] = typ_tid

def split_label_ftag_trace(item, ftags, nid, trace_refs, trace_eqs):
    if '=' in item:
        item, eq = item.split('=')
        if (i := eq.find('-')) >= 0:
            item += eq[i:]
            eq = eq[:i]
        assert eq.isdigit()
        trace_eqs[nid] = eq
    if '-' in item:
        segments = item.split('-')
        item = segments.pop(0)
        if segments[-1].isdigit(): # for both '-='
            trace_refs[nid] = segments.pop()
        if segments:
            ftags[nid] = '-'.join(segments)
    return item

def read_materials(tree, trace_types, *catch_nodes, **wtnt_fns):
    bottom = {}
    top_down = {}
    ftags = {}
    trace_refs = {} # reference
    trace_eqs = {}
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
            init_trace_id(trace_types, wd, tg, trace_ids, nid)
        elif status == _CMD_BOL: # item is a tag or a label
            cnid = max(short_memory[item]) # cnid never belongs to item
            short_memory[item].remove(cnid)
            if not short_memory[item]:
                short_memory.pop(item)
            top_down[nid].append(cnid) # cnid belongs to nid
        elif status == _CMD_EOL:
            # item is the parent label
            short_memory[item] |= short_memory.pop(None)
            item = split_label_ftag_trace(item, ftags, nid, trace_refs, trace_eqs)
            if item in catch_nodes:
                catch_nodes[item].add(nid)

            children = {}
            for cnid in top_down[nid]:
                children[cnid] = ftags.pop(cnid, None)

                if cnid in trace_refs: # trace reference with pid/cid
                    tid = trace_refs.pop(cnid)
                    trace_refs[tid] = TraceRef(nid, cnid, *boundary(top_down, cnid))
                    # if tid not in trace_refs or not top_down[trace_refs[tid].cid].label.startswith('WH'): # wh-movement has the priority
                        # if tid in trace_refs:
                        #     breakpoint()
                        #     print(tid)

                if cnid in trace_ids: # trace identity with pid/cid/bid
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
    return bottom, top_down, nid, nt_start, get_inode, trace_refs, trace_ids, trace_eqs, catch_nodes
    
from utils.param_ops import get_sole_key, change_key, shift_key
def maintain(bottom_info, top_down, root_id, nt_start, get_inode, type_count = None):
    nids = {root_id}
    bottom_up = {}
    remove_nids = set()
    seen_nids = set()
    while nids:
        cids = set()
        for nid in nids:
            if nid < nt_start: # no redundant bids
                if bottom_info[nid][1] == '-NONE-':
                    remove_nids.add(nid)
            else:
                if children := top_down[nid].children:
                    for cid in children:
                        cids.add(cid); bottom_up[cid] = nid
                else:
                    remove_nids.add(nid)
                seen_nids.add(nid)
        nids = cids
    for cid in remove_nids:
        if cid < nt_start:
            bottom_info.pop(cid)
        else:
            top_down.pop(cid) # maintain
        pid = bottom_up.pop(cid)
        top_down[pid].children.pop(cid)
        while not top_down[pid].children: # empty again
            top_down.pop(pid)
            cid = pid
            pid = bottom_up.pop(cid)
            top_down[pid].children.pop(cid) # maintain
    for nid in top_down.keys() - seen_nids:
        top_down.pop(nid)

    vroot = top_down.pop(root_id)
    assert vroot.label == C_VROOT
    root_id = get_sole_key(vroot.children)

    new_top_down = {}
    condense = {bid: eid for eid, bid in enumerate(sorted(bottom_info), 1) if eid != bid}
    bottom = [(condense.get(bid, bid), w, t) for bid, (w, t) in bottom_info.items()]
    for nid, td in top_down.items():
        children = {}
        for cid, ftag in td.children.items():
            children[get_inode(condense.get(cid, cid))] = ftag
        nid = 0 if nid == root_id else get_inode(nid)
        new_top_down[nid] = TopDown(td.label, children)
    if type_count is not None:
        new_type_count = {}
        for cid, pt in type_count.items():
            cid = get_inode(condense.get(cid, cid))
            new_type_count[cid] = {(0 if p == root_id else get_inode(p)): t for p, t in pt.items()}
        return bottom, new_top_down, new_type_count
    return bottom, new_top_down

def read_tree(tree, *,
              adjust_fn = fix_for_dptb,
              return_type_count = False):
    wtnt_fns = {}
    if callable(adjust_fn):
        wtnt_fns.update(adjust_fn(tree))

    (bottom, top_down, root_id, nt_start, get_inode, trace_refs, trace_ids, _,
     catch_nodes) = read_materials(tree, E_DISCO, 'PRN', **wtnt_fns)
    remaining_PRN_nodes = catch_nodes.pop('PRN')

    # cross trace along the bottom (ordered and reversed for bottom.pop(i) stability)
    c2p_history = {}
    type_count = defaultdict(int)
    for typ, tid, i_pid, i_cid, i_bid, r_pid, r_cid, lhs, rhs in tree_trace_gen(trace_refs, trace_ids):
        i_pid = c2p_history.pop(i_cid, i_pid) # to relocate attachment ... (3 chains)
        s_ftag = top_down[r_pid].children.pop(r_cid)
        d_ftag = top_down[i_pid].children.pop(i_cid)
        v_wd, v_tg = bottom.pop(i_bid)
        assert v_wd.endswith(tid)
        assert v_tg == '-NONE-'
        if s_ftag and d_ftag:
            ftag = s_ftag if s_ftag == d_ftag else (s_ftag + ':' + d_ftag)
        else:
            ftag = s_ftag or d_ftag
        top_down[i_pid].children[r_cid] = ftag
        c2p_history[r_cid] = i_pid # add r_cid as a i_pid child
        if lhs <= i_bid <= rhs: # PRN
            r_cid, s_ccid = PathFinder(top_down, r_cid).find_labeled_on_path('PRN', i_pid)
            ftag = top_down[r_cid].children.pop(s_ccid)
            top_down[r_pid].children[s_ccid] = ftag
            remaining_PRN_nodes.remove(s_ccid)
            typ += '-PRN'
        type_count[typ] += 1

    bottom, top_down = maintain(bottom, top_down, root_id, nt_start, get_inode)

    if return_type_count:
        return bottom, top_down, type_count, trace_refs, trace_ids
    return bottom, top_down

def raise_eq(label):
    assert '=' not in label
    return label

def fix_for_gptb(tree):
    fix_for_ptb(tree, set())
    remove_irrelevant_trace(tree, E_GRAPH)
    return dict(w_fn = XRB2brackets, nt_fn = do_nothing)

def graph_trace_gen(trace_refs, trace_ids):
    priority, loops = trace_priority(trace_refs, trace_ids)
    assert not loops
    # stage = 0
    # print(groups)
    # for is_loop, chain in sorted(groups, key = lambda x: len(x.chain)):
    #     assert not is_loop
    #     for tid in chain:
    #         priority[tid] += stage
    #     stage += len(chain)
    # print(priority)
    # select the nearest for multi-attachment
    for tid in sorted(priority, key = priority.get, reverse = True):
        # print(tid)
        yield trace_refs.pop(tid), trace_ids.pop(tid)

def read_graph(tree, *,
               trace_types = E_GRAPH,
               adjust_fn = fix_for_gptb,
               return_type_count = False):
    wtnt_fns = {}
    if callable(adjust_fn):
        wtnt_fns.update(adjust_fn(tree))

    (bottom, top_down, root_id, nt_start, get_inode, trace_refs, trace_ids, trace_eqs,
     catch_nodes) = read_materials(tree, trace_types, 'PRN', **wtnt_fns)
    remaining_PRN_nodes = catch_nodes.pop('PRN')

    c2p_history = {}
    c2ps_history = {}
    type_count = defaultdict(dict)
    check_combine = defaultdict(list)
    eq_template_refs = {}
    eq_template_maps = {}
    eq_tids = set(trace_eqs.values())
    for (r_pid, r_cid, lhs, rhs), tids in graph_trace_gen(trace_refs, trace_ids):
        parents = {}
        ref_children = top_down[r_pid].children
        # print('Ref:', r_pid, r_cid)
        for typ, tid, i_pid, i_cid, i_bid in tids:
            if i_cid in c2p_history:
                # print('    ', i_pid)
                i_pid = c2p_history.pop(i_cid)
            # print(f'{typ}-{tid}:', i_pid, i_cid, i_bid)
            d_ftag = top_down[i_pid].children.pop(i_cid)
            v_wd, v_tg = bottom.pop(i_bid)
            assert v_wd.endswith(tid)
            assert v_tg == '-NONE-'
            if typ in A_MOVEMENT:
                parents[i_cid] = i_pid, typ
                s_ftag = None # should not propagate for new attachment
            else:
                s_ftag = ref_children[r_cid]
                type_count[r_cid][i_pid] = typ
            if s_ftag and d_ftag:
                ftag = s_ftag if s_ftag == d_ftag else (s_ftag + ':' + d_ftag)
            else:
                ftag = s_ftag or d_ftag
            if i_cid in c2ps_history:
                top_down[i_pid].children[i_cid] = d_ftag
                for m_pid in c2ps_history[i_cid]:
                    change_key(top_down[m_pid].children, i_cid, r_cid)
                    # print('change key child of', m_pid, 'from', i_cid, 'to', r_cid)
            else:
                top_down[i_pid].children[r_cid] = ftag
                if ii_pid := c2p_history.pop(r_cid, None):
                    check_combine[r_cid].append(ii_pid)
                c2p_history[r_cid] = i_pid # add r_cid as a i_pid child
            # print('MA:', mattached_c2ps)
            eq_template_maps[i_cid] = r_cid
            if lhs <= i_bid <= rhs:
                p_pid, p_cid = PathFinder(top_down, r_cid).find_labeled_on_path('PRN', i_pid)
                # print('RPN', p_cid, 'from', p_pid, 'to', r_pid)
                ftag = top_down[p_pid].children.pop(p_cid)
                top_down[r_pid].children[p_cid] = ftag
                remaining_PRN_nodes.remove(p_cid)
                type_count[r_cid][i_pid] += '-PRN'
        if parents:
            flatten = {r_pid: typ}
            # print(c2ps_history)
            for i_cid, (i_pid, typ) in parents.items():
                if i_cid in c2ps_history:
                    flatten.update(c2ps_history.pop(i_cid))
                else:
                    flatten[i_pid] = typ
            # print('To flatten', flatten)
            c2ps_history[r_cid] = flatten
            if len(bids := top_down[r_cid].children) > 1 or \
                (bt := bottom.get(get_sole_key(bids))) and bt[1] != '-NONE-':
                for i_pid, typ in flatten.items():
                    type_count[r_cid][i_pid] = (typ + '0') if i_pid == r_pid else typ
        else:
            ref_children.pop(r_cid)
        if tid in eq_tids:
            eq_template_refs[r_cid] = tid
        
    if len(multi_prn := top_down[root_id].children) > 1:
        coordinated_prn = {}
        for nid in multi_prn.copy():
            assert top_down[nid].label == 'PRN'
            coordinated_prn[nid] = multi_prn.pop(nid)
        coord_nid = root_id + 1
        multi_prn[coord_nid] = None
        top_down[coord_nid] = TopDown('PRN', coordinated_prn)
        for r_cid, i_pids in check_combine.items():
            ii_pid = c2p_history.pop(r_cid)
            for i_pid in i_pids:
                assert top_down.pop(i_pid) == top_down[ii_pid]
                for td in top_down.values():
                    if i_pid in td.children:
                        change_key(td.children, i_pid, ii_pid)

    if trace_eqs:
        paths = {}
        path_finder = PathFinder(top_down, root_id)
        for nid, tid in trace_eqs.items():
            while nid in eq_template_maps:
                nid = eq_template_maps[nid]
            for path in path_finder[nid]:
                paths[(True, path)] = tid
            if tid in trace_refs:
                eq_template_refs[trace_refs.pop(tid).cid] = tid
        for nid, tid in eq_template_refs.items():
            while nid in eq_template_maps:
                nid = eq_template_maps[nid]
            for path in path_finder[nid]:
                paths[(False, path)] = tid

        try:
            eq, template_parent = common_coordination(top_down, paths)
        except ValueError:
            eq, template_parent = special_coordination(top_down, paths)
        template_cids = set()
        template_paths = []
        imitator_paths = {}
        for path in sorted(eq, key = lambda x: len(x)):
            if len(path) < 2: continue
            pid = path[0]
            tid = eq[path]
            if pid == template_parent:
                template_cids.add(path[1])
                template_paths.append((path[1:], tid, all(tid != ptm[1] for ptm in template_paths)))
            elif pid not in imitator_paths:
                imitator_paths[pid] = {tid: path[1:]}
            elif tid not in imitator_paths[pid]:
                imitator_paths[pid][tid] = path[1:]
        eq = imitator_paths # rename for my own readability
        template_ma = {}
        for c,f in top_down[template_parent].children.items():
            if c not in template_cids and is_necessary(bottom, top_down, template_parent, c, f):
                type_count[c][template_parent] = 'C0'
                template_ma[c] = f
        for imitator_parent, imitator_paths in eq.items():
            for t_cid, t_ftag in template_ma.items():
                type_count[t_cid][imitator_parent] = 'C'
                top_down[imitator_parent].children[t_cid] = t_ftag
            imitate(top_down, type_count,
                template_parent, (ptm for ptm in template_paths if ptm[1] in imitator_paths),
                imitator_parent, imitator_paths, {})

    bottom, top_down, type_count = maintain(bottom, top_down, root_id, nt_start, get_inode, type_count)
    
    if return_type_count:
        return bottom, top_down, type_count, trace_refs, trace_ids
    return bottom, top_down

def label_checker(label):
    if label == 'S':
        return lambda x: x.startswith(label)
    return lambda x: x == label

__pc = {'ADVP': ('RB',), 'PP': ('IN', 'TO') }
def is_necessary(bottom, top_down, pid, cid, ftag):
    assert cid in top_down[pid].children
    pl = top_down[pid].label
    if cid in top_down:
        cl = top_down[cid].label
    else:
        _, cl = bottom[cid]
    if pl.startswith('S'):
        return cl.startswith('V')
    elif pl.startswith('V'):
        return cl.startswith('V') or cl.startswith('N') and ftag in (None, 'PRD')
    elif pl.startswith('N'):
        return cl.startswith('N')
    elif pl in __pc:
        return cl in __pc[pl]
    elif pl in ('FRAG',):
        return False
    raise ValueError(f'Unsupported {pl}')

def get_common_path(i_paths):
    if len(i_paths) == 1:
        return get_sole_key(i_paths)[:-1]
    assert i_paths, 'Not found traces (=)'
    common_path = []
    j_paths = {p: 0 for p in i_paths}
    safe = True
    while safe:
        sole = defaultdict(dict)
        for path, depth in j_paths.items():
            if depth < len(path):
                j_paths[path] += 1
                sole[path[depth]][path] = False
            elif safe:
                safe = False
        if safe:
            if len(sole) == 1:
                common_path.extend(sole)
                continue
            x_main = None
            for x in sole:
                for y, y_paths in sole.items():
                    if x != y:
                        for y_path in y_paths:
                            if x in y_path:
                                if x_main is None:
                                    x_main = x
                                elif x_main != x:
                                    safe = False
                                    break
                                y_paths[y_path] = True
                                j_paths[y_path] = y_path.index(x) + 1
                    if not safe: break
                if not safe: break
            safe &= x_main is not None
            safe &= all(not any(paths.values()) if x == x_main else all(paths.values()) for x, paths in sole.items())
            if safe:
                j_paths = {path:j_paths[path] for path in sole.pop(x_main)}
                for paths in sole.values():
                    for path in paths:
                        i_paths.pop(path)
                common_path.append(x_main)
    return tuple(common_path)

def common_coordination(top_down, paths):
    i_paths = {path: eq for (eq, path) in paths if (not eq, path) not in paths}
    common_path = get_common_path(i_paths)

    test_depth = len(common_path)
    while test_depth > 0:
        test_nid = common_path[test_depth - 1]
        test_label = label_checker(top_down[test_nid].label)
        if all(test_label(top_down[i[test_depth]].label) for i in i_paths):
            template = set()
            coord_pids = set()
            for path, eq in i_paths.items():
                pid = path[test_depth]
                coord_pids.add(pid)
                if not eq: template.add(pid)
            if template:
                template = get_sole_key(template)
            else: # TODO detect errors / result multi-template
                for pid in top_down[test_nid].children:
                    if pid in top_down and pid not in coord_pids and test_label(top_down[pid].label):
                        assert not template, f'Multiple templates {template} & {pid}'
                        template = pid
            irrelevant = set()
            coord_paths = {}
            for (eq, path), tid in paths.items():
                if common_path[:test_depth] == path[:test_depth] and \
                    eq ^ (template == path[test_depth]):
                    coord_paths[path[test_depth:]] = tid
                else:
                    irrelevant.add((eq, path))
            for key in irrelevant:
                paths.pop(key)
            return coord_paths, template
        test_depth -= 1
    raise ValueError(f'Coordination not found with {len(top_down)} td and {len(paths)} eq.paths')

def special_coordination(top_down, paths):
    coord_paths = {}
    imitator = {}
    template = {}
    for (eq, path), tid in paths.items():
        if eq:
            imitator[path] = tid
        else:
            template[path] = tid
    assert len(imitator.values()) == len(set(imitator.values()))
    t_path = []
    for elems in zip(*template):
        if len(sole := set(elems)) == 1:
            t_path.extend(sole)
        else: break
    t_depth = len(t_path) - 1
    for path, tid in template.items():
        coord_paths[path[t_depth:]] = tid
    template = t_path[t_depth]
    label = top_down[template].label
    i_path = []
    for elems in zip(*imitator):
        if len(sole := set(elems)) == 1:
            i_path.extend(sole)
        else: break
    i_depth = len(i_path) - 1
    for path, tid in imitator.items():
        coord_paths[path[i_depth:]] = tid
        assert top_down[path[i_depth]].label == label
    return coord_paths, template

def imitate(top_down, type_count, t_pid, t_paths, i_pid, i_paths, t2i_construction): # troublesome default {}
    new_pids = {}
    new_t_paths = []
    new_i_paths = defaultdict(list)
    new_shifts = {}
    t_children = top_down[t_pid].children
    i_children = top_down[i_pid].children
    i_tids = set()
    new_i_cid = max(top_down) + 1
    for t_path, tid, t_main in t_paths:
        i_path = i_paths[tid] # TODO jit
        i_tids.add(tid)
        t_len = len(t_path)
        i_len = len(i_path)
        if t_len == 1:
            assert i_len == 1
            i_nid, = i_path
            t_nid, = t_path
            # assert top_down[t_nid].label == top_down[i_nid].label, 'Transfer between different label'
            assert t_main ^ (t_nid in t2i_construction)
            if i_nid in i_children: # MA
                if i_nid in type_count: # otherwise, a creation
                    type_count[i_nid][i_pid] += 'C'
            else:
                i_children[i_nid] = t_children[t_nid]
            t2i_construction[t_nid] = i_nid
        else:
            t_cid = t_path[0]
            i_cid = i_path[0]
            if t_len < 3 or t_path[1] not in t2i_construction:
                new_t_paths.append((t_path, tid, t_main))
            if t_len == i_len:
                assert top_down[t_cid].label == top_down[i_cid].label, 'Transfer between different label'
                # assert False, 'Reduce together (TODO)'
                new_i_paths[tid] = i_path[1:]
                new_pids[t_cid] = i_cid
            else:
                if tid not in new_i_paths:
                    new_i_paths[tid] = i_path
                if t_cid in new_shifts:
                    _, t2i_shifts = new_shifts[t_cid]
                else:
                    t2i_shifts = defaultdict(set)
                    # if t_len == 2 and top_down[t_cid].label == top_down[i_cid].label:
                    if t_cid in t2i_construction:
                        dst_i_cid = t2i_construction[t_cid] # parent of i_cid
                    else:
                        new_pids[t_cid] = dst_i_cid = new_i_cid
                        new_i_cid += 1
                    new_shifts[t_cid] = dst_i_cid, t2i_shifts
                t2i_shifts[t_path[1]].add(i_cid if t_main else None)
                
    for tid in i_tids - i_paths.keys():
        for i_path in i_paths[tid]:
            i_cid, = i_path
            assert i_cid in i_children

    for t_cid, (dst_i_cid, t2i_shift) in new_shifts.items():
        if creation := (dst_i_cid not in top_down):
            top_down[dst_i_cid] = TopDown(top_down[t_cid].label, new_children := {})
            i_children[dst_i_cid] = t_children[t_cid] # take root by imitate/copy
            type_count[dst_i_cid][i_pid] = 'CC'
        else:
            new_children = top_down[dst_i_cid].children
        for t_ccid, ftag in top_down[t_cid].children.items():
            if t_ccid in t2i_construction:
                if (i_ccid := t2i_construction[t_ccid]) not in new_children:
                    new_children[i_ccid] = ftag
                if dst_i_cid in type_count[i_ccid]:
                    type_count[i_ccid][dst_i_cid] += 'C'
                else:
                    type_count[i_ccid][dst_i_cid] = 'C'
            elif t_ccid in t2i_shift:
                # t_main is not None and up root i_cid from i_pid
                # original ftag first, then template's ftag
                for i_cid in t2i_shift[t_ccid]:
                    if i_cid is None:
                        continue
                    shift_key(i_cid, i_children, new_children)
            elif creation:
                if t_ccid not in type_count:
                    type_count[t_ccid][t_cid] = 'C0'
                type_count[t_ccid][dst_i_cid] = 'C'
                new_children[t_ccid] = ftag # copy template
        if creation:
            t2i_construction[t_cid] = dst_i_cid
            
    for new_t_pid, new_i_pid in new_pids.items():
        sub_t_paths = []
        sub_i_paths = {}
        for t_path, tid, t_main in new_t_paths:
            if new_t_pid == t_path[0]:
                sub_t_paths.append((t_path[1:], tid, t_main))
                sub_i_paths[tid] = new_i_paths[tid]
        imitate(top_down, type_count, new_t_pid, sub_t_paths, new_i_pid, sub_i_paths, t2i_construction)