from data.cross import TopDown
from data.cross import defaultdict, gap_degree, height_gen
from data.cross import _dep_on, _dep_combine
from random import random
from utils.types import F_RANDOM, F_LEFT, F_RIGHT, F_DEP, F_CON

def _closest(order, length):
    if isinstance(order, float):
        order = round(order * (length - 1e-10) - 0.5)
    return order

def _find_dep(children, dep, node, bt_head):
    for _node in children:
        if _node == node:
            continue
        if _node == bt_head or _node in dep and _dep_on(dep, node, _node):
            return _node

def _multi_hash(bottom, top_down, factor):
    # make a new td: including head flag
    bottom_flag = [True for _ in bottom]
    bottom_trace = [None for _ in bottom]
    p_head = {}
    for p_node, td in top_down.items():
        location = []
        children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                children.append(node)
                bottom_trace[nid] = node
                bottom_flag[nid] = False

        if factor == F_RANDOM:
            loc = random()
        elif factor == F_LEFT:
            loc = 0
        elif factor == F_RIGHT:
            loc = -1
        else:
            raise NotImplementedError()
        cid = _closest(loc, len(children))
        p_head[children[cid]] = p_node, None, location
    return bottom_trace, p_head

I_EXH = 1
I_LPH = 2
def _dep_hash(bottom, top_down, dependency):
    # make a new td: including head flag
    bottom_flag = [True for _ in bottom]
    bottom_trace = [None for _ in bottom]
    p_head = {}
    issues = {}
    for p_node, td in top_down.items():
        location = []
        children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                children.append(node)
                bottom_trace[nid] = node
                bottom_flag[nid] = False

        loc_in = {}
        dep_pt = {}
        external_head = defaultdict(set)
        for nid, node in enumerate(children):
            loc_in[node] = nid
            bt_head = dependency[node].label
            if bt_head is None: continue # ROOT
            _node = _find_dep(children, dependency, node, bt_head)
            if _node is None:
                external_head[bt_head].add(node)
            else:
                dep_pt[node] = _node
        p_issue = {}
        if dep_pt:
            dep_in = defaultdict(int)
            for node in children:
                anti_loop = set({node})
                while head := dep_pt.get(node):
                    dep_in[head] += 1; node = head
                    if node in anti_loop: p_issue[I_LPH] = dep_pt; break
                    else: anti_loop.add(node)
            head = max(dep_in, key = dep_in.get)
            loc = loc_in[head]
        else:
            loc = _closest(random(), len(children))
            head = children[loc]
            if len(external_head) != 1: p_issue[I_EXH] = external_head
            # if len(external_head) != 1: # DPTB need both this and anti_loop
            #     print(external_head)
            #     breakpoint()
        _dep_combine(dependency, head, p_node, *(node for node in children if node != head))

        cid = _closest(loc, len(children))
        p_head[children[cid]] = p_node, None, location
        if p_issue: issues[p_node] = p_issue
    return bottom_trace, p_head, issues

def _continuous_hash(bottom, bottom_top_down, future_top_down, gaps, boundaries, bottom_up, factor, bottom_ref):
    # make a new td: including head flag
    bottom_flag = [True for _ in bottom]
    bottom_trace = [None for _ in bottom]
    p_head = {}
    for p_node, td in sorted(bottom_top_down.items(), key = lambda x: gaps[x[0]]):
        location = []
        children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                children.append(node)
                bottom_trace[nid] = node
                bottom_flag[nid] = False

        factor = random() if factor is None else factor
        if gaps[p_node] == 0: # 0 does not effect order in the next bottom
            node = children[_closest(factor, len(children))]
            p_head[node] = p_node, bottom_ref[node], location
        else:
            continuous = []
            pp_node = bottom_up[p_node]
            further_continuous = []
            if pp_node in bottom_up:
                for pc_node in future_top_down[pp_node].children:
                    if pc_node == p_node:
                        continue
                    if pc_node in boundaries:
                        pc_lhs, pc_rhs = boundaries[pc_node]
                    else:
                        pc_lhs = pc_rhs = bottom_ref[pc_node]
                    for cid, node in enumerate(children):
                        b_head = bottom_ref[node]
                        if b_head < pc_lhs:
                            if any(b_head < b < pc_lhs and n not in children for n, b in bottom_ref.items()):
                                further_continuous.append(cid)
                            else:
                                # import pdb; pdb.set_trace()
                                continuous.append(cid)
                        elif pc_rhs < b_head:
                            if any(pc_rhs < b < b_head and n not in children for n, b in bottom_ref.items()):
                                further_continuous.append(cid)
                            else:
                                # import pdb; pdb.set_trace()
                                continuous.append(cid)
            if continuous:
                children = [children[cid] for cid in set(continuous)]
            elif further_continuous:
                further_continuous = set(further_continuous)
                wrong_houses = defaultdict(int)
                while pp_node in bottom_up:
                    ppp_node = bottom_up[pp_node]
                    for ppc_node in future_top_down[ppp_node].children:
                        if ppc_node == pp_node or ppc_node not in boundaries:
                            continue
                        ppc_lhs, ppc_rhs = boundaries[ppc_node]
                        for cid in further_continuous:
                            if ppc_lhs < bottom_ref[children[cid]] < ppc_rhs:
                                wrong_houses[cid] += 1
                    pp_node = ppp_node

                while wrong_houses and further_continuous == wrong_houses.keys():
                    lightest_toll = min(wrong_houses.values())
                    wrong_houses = {k:v for k,v in wrong_houses.items() if v > lightest_toll}

                further_continuous -= wrong_houses.keys()
                if further_continuous:
                    # if wrong_houses:
                    #     print(further_continuous, bottom_top_down)
                    #     print(children)
                    #     print(wrong_houses)
                    children = [children[cid] for cid in further_continuous]
                    # if wrong_houses:
                    #     print(children)
                    #     # try:
                    #     #     node = children[_closest(factor, len(children))]
                    #     # except:
                    #     import pdb; pdb.set_trace()

            node = children[_closest(factor, len(children))]
            p_head[node] = p_node, bottom_ref[node], location
    return bottom_trace, p_head

def _is_disc(locations):
    return any(lhs + 1 != rhs for lhs, rhs in zip(locations, locations[1:]))

def cross_signals(bottom, node2tag, bottom_unary, top_down, factor, # float:midin :most_continuous :random :dep
                  l2i = None,
                  dependency_or_mid_in = None,
                  verbose_file = None,
                  pos_prefix = '#'):
    # sub_suffix = '.'
    f_continuous = factor == F_CON
    f_dep = factor == F_DEP
    if f_continuous:
        mid_in = dependency_or_mid_in
        boundaries = {}
        bottom_ref = {node: i for i, node in enumerate(bottom)}
        bottom_up = {}
        gaps = gap_degree(bottom, top_down, None, True)
    elif f_dep and verbose_file:
        dep_issues = []

    top_down_group_by_height = [] # ordered by td dependency
    for node, height in height_gen(top_down, 0):
        td = top_down[node]
        if f_continuous:
            cb = []
            for cid in td.children:
                if cid in boundaries:
                    cb += boundaries[cid]
                else:
                    cb.append(bottom_ref[cid])
                bottom_up[cid] = node
            boundaries[node] = min(cb), max(cb)
        if height == len(top_down_group_by_height):
            top_down_group_by_height.append({node: td})
        else:
            top_down_group_by_height[height][node] = td

    layers_of_label = []
    layers_of_space = []
    layers_of_disco = []
    while len(bottom) > 1:
        label_layer = []
        for node in bottom:
            if node in bottom_unary:
                label = bottom_unary[node]
            elif node in node2tag:
                label = pos_prefix + node2tag[node]
            else:
                label = top_down[node].label
            label_layer.append(l2i(label) if l2i else label)

        bottom_top_down = top_down_group_by_height.pop(0)
        if f_continuous:
            future_top_down = {}
            for td in top_down_group_by_height:
                future_top_down.update(td)
            bottom_trace, p_head = _continuous_hash(bottom, bottom_top_down, future_top_down, gaps, boundaries, bottom_up, mid_in, bottom_ref)
        elif f_dep:
            bottom_trace, p_head, issues = _dep_hash(bottom, bottom_top_down, dependency_or_mid_in)
            if verbose_file:
                if issues:
                    line = f'#{len(dep_issues)}: '
                    for e_nid, _issues in issues.items():
                        line +=  f'{top_down[e_nid].label} '
                        for ety, _issue in _issues.items():
                            if ety == I_EXH:
                                line += '[E '
                                line += '; '.join('(' + ', '.join(top_down[n].label if n in top_down else n for n in ns) + f') -> {h}' for h, ns in _issue.items())
                                line +=  ']'
                            else:
                                line += '[L '
                                line += ', '.join(f'{n} -> {h}' for n, h in _issues.items())
                                line += ']'
                else:
                    line = None
                dep_issues.append(line)
        else:
            bottom_trace, p_head = _multi_hash(bottom, bottom_top_down, factor)
        new_bottom = []
        space_layer = [None for _ in bottom]
        counter = 0
        disco_set = {}
        
        for bid, t in enumerate(bottom_trace):
            if t is None:
                new_bottom.append(bottom[bid])
                space_layer[bid] = counter
                counter += 1
            elif t in p_head:
                p_node, b_head, children = p_head.pop(t)
                new_bottom.append(p_node)
                for t in children:
                    space_layer[t] = counter
                if _is_disc(children):
                    disco_set[counter] = children
                if f_continuous:
                    bottom_ref[p_node] = b_head
                    for t in children:
                        bottom_ref.pop(bottom[t])
                counter += 1

        bottom = new_bottom
        layers_of_label.append(label_layer)
        layers_of_space.append(space_layer)
        layers_of_disco.append(disco_set)
    
    assert len(bottom) == 1 and not top_down_group_by_height
    node = bottom[0]
    if top_down:
        label = top_down[node].label
    elif bottom_unary:
        label = bottom_unary[node]
    else:
        assert 'This should not happen in a treebank'
        label = pos_prefix + node2tag[node]
    if l2i is not None:
        label = l2i(label)
    layers_of_label.append([label])
    if verbose_file and f_dep and any(dep_issues):
        headline, file, lines = verbose_file
        file.write(headline + '\n' + lines + '\n')
        for line in dep_issues:
            if line: file.write(line)
        file.write('\n')

    return layers_of_label, layers_of_space, layers_of_disco


from data.cross import E_SHP, E_CMB, E_LBL, _combine, bottom_trees
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_space,
               fallback_label = None,
               layers_of_weight     = None,
               perserve_sub         = False):

    (NTS, bottom_len, track_nodes, terminals, non_terminals, top_down, track_fall_back,
     error_layer_id) = bottom_trees(word, bottom_tag, layers_of_label, fallback_label, perserve_sub)

    add_weight_base = layers_of_weight is not None
    weight_nodes = {}
    if add_weight_base:
        headedness_stat = {}
        nid2tag = {n: tg for (n, _, tg) in terminals}
    else:
        headedness_stat = None

    if track_fall_back:
        def fallback(NTS):
            for pid in track_nodes:
                if pid in non_terminals and non_terminals[pid][0] in '_#':
                    non_terminals.pop(pid)
                    top_down[NTS].update(top_down.pop(pid))
                else:
                    top_down[NTS].add(pid)
            non_terminals[NTS] = fallback_label
            return NTS - 1

    for lid, space_layer in enumerate(layers_of_space):
        if track_fall_back:
            next_layer_size = len(layers_of_label[lid + 1])
            if set(range(next_layer_size)) != set(layers_of_space[lid]):
                error_layer_id = lid, bottom_len, E_LBL
                if next_layer_size == 1:
                    space_layer = [0 for x in space_layer]
                else:
                    NTS = fallback(NTS)
                    break

        td = defaultdict(list)
        for src, dst in enumerate(space_layer):
            td[dst].append(src)
        td = sorted(td.items(), key = lambda pc: pc[0])
        add_weight = add_weight_base and lid < len(layers_of_weight)
        
        combined = []
        new_track_nodes = []
        for pid, cids in td:
            if len(cids) > 1:
                labels = layers_of_label[lid + 1][pid]
                if add_weight:
                    max_weight = 0
                    np_leaves = []
                    for cid in cids:
                        track_nid = track_nodes[cid]
                        if (weight := layers_of_weight[lid][cid, 0]) > max_weight:
                            max_weight = weight
                            head_label = (nid2tag, non_terminals)[track_nid in non_terminals][track_nid]
                        if 'NP' in labels and track_nid in nid2tag:
                            np_leaves.append(nid2tag[track_nid])
                        non_terminals[NTS] = f'{weight * 100:.0f}%'
                        top_down[NTS].add(track_nid)
                        weight_nodes[NTS] = lid
                        track_nodes[cid] = NTS
                        NTS -= 1
                    if head_label != 'DT' and 'DT' in np_leaves:
                        head_label += '*'
                    if labels in headedness_stat:
                        label_cnt, head_cnts = headedness_stat[labels]
                    else:
                        label_cnt = 0
                        head_cnts = defaultdict(int)
                    head_cnts[head_label] += 1
                    headedness_stat[labels] = label_cnt + 1, head_cnts
                        
                labels = [labels] if perserve_sub or labels[0] in '#_' else labels.split('+')
                non_terminals[NTS] = labels.pop()
                # >j<
                for cid in cids:
                    _combine(NTS, track_nodes[cid], non_terminals, top_down, perserve_sub)
                while labels: # unary DIY
                    NTS -= 1
                    non_terminals[NTS] = labels.pop()
                    top_down[NTS] = set({NTS + 1})
                new_track_nodes.append(NTS)
                NTS -= 1
                combined.extend(cids)
            else:
                new_track_nodes.append(track_nodes[cids.pop()])

        # if isinstance(fallback_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and new_track_nodes == track_nodes and track_fall_back: # no action is taken
            error_layer_id = lid, bottom_len, E_CMB
            NTS = fallback(NTS)
            break

        track_nodes = new_track_nodes

    if not error_layer_id:
        if not non_terminals:
            assert fallback_label, 'should provide a fallback root label'
            top_down[NTS].update(tid for tid, _, _ in terminals)
            non_terminals[NTS] = fallback_label
            NTS -= 1
            error_layer_id = -1, bottom_len, E_LBL

    # import pdb; pdb.set_trace()
    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, {x: None for x in top_down.pop(nid)})
    top_down[0] = top_down.pop(NTS + 1)

    if add_weight_base:
        return terminals, top_down, error_layer_id, weight_nodes, headedness_stat
    return terminals, top_down, error_layer_id

def continuous_fence(space_layer, disco_set):
    count = 0
    split_layer = []
    for lhs, rhs in zip(space_layer, space_layer[1:] + [-1]):
        if lhs in disco_set:
            continue
        else:
            count += 1
        if lhs != rhs:
            split_layer.append(count)
    if split_layer:
        split_layer.insert(0, 0)
    return count, split_layer

def total_fence(space_layer):
    count = len(space_layer)
    space_layer = [-1] + space_layer + [-1]
    split_layer = []
    for sid, (lhs, rhs) in enumerate(zip(space_layer, space_layer[1:])):
        if lhs != rhs:
            split_layer.append(sid)
    return count, split_layer