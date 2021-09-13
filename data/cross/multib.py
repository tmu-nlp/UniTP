from data.cross import has_multiple, TopDown, _read_dpenn, _read_graph, _pre_proc
from data.cross import defaultdict, gap_degree, height_gen, add_efficient_subs
from data.cross import _new_dep, _dep_n_prefix, _dep_on, _dep_combine
from utils.param_ops import replace_args_kwargs
from random import random

F_RANDOM = 'random'
F_LEFT = 'left'
F_RIGHT = 'right'
F_DEP = 'head'
F_CON = 'continuous'
E_FACTOR = F_RANDOM, F_LEFT, F_RIGHT, F_DEP, F_CON

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
            if pp_node in bottom_up: # not root_id
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

def cross_signals(bottom, node2tag, bottom_unary, top_down, root_id, factor, # float:midin :most_continuous :random :dep
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
        gaps = gap_degree(bottom, top_down, root_id)
    elif f_dep and verbose_file:
        dep_issues = []

    top_down_group_by_height = [] # ordered by td dependency
    for node, height in height_gen(top_down, root_id):
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
    
    node = bottom.pop()
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
    assert not bottom and not top_down_group_by_height
    if verbose_file and f_dep and any(dep_issues):
        headline, file, lines = verbose_file
        file.write(headline + '\n' + lines + '\n')
        for line in dep_issues:
            if line: file.write(line)
        file.write('\n')

    return layers_of_label, layers_of_space, layers_of_disco


from data.cross import E_SHP, E_CMB, E_LBL, _combine, draw_str_lines, gap_degree, bottom_trees
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_space,
               fall_back_root_label = None,
               layers_of_weight     = None,
               perserve_sub         = False):

    (NTS, bottom_len, track_nodes, terminals, non_terminals, top_down, track_fall_back,
     error_layer_id) = bottom_trees(word, bottom_tag, layers_of_label, fall_back_root_label, perserve_sub)
    non_terminal_end = NTS

    add_weight_base = layers_of_weight is not None
    weight_nodes = {}

    if track_fall_back:
        def fallback(non_terminal_end):
            for pid in track_nodes:
                if pid in non_terminals and non_terminals[pid][0] in '_#':
                    non_terminals.pop(pid)
                    top_down[non_terminal_end].update(top_down.pop(pid))
                else:
                    top_down[non_terminal_end].add(pid)
            non_terminals[non_terminal_end] = fall_back_root_label
            return non_terminal_end + 1

    for lid, space_layer in enumerate(layers_of_space):
        if track_fall_back:
            next_layer_size = len(layers_of_label[lid + 1])
            if set(range(next_layer_size)) != set(layers_of_space[lid]):
                # import pdb; pdb.set_trace()
                error_layer_id = lid, E_LBL, bottom_len
                if next_layer_size == 1:
                    space_layer = [0 for x in space_layer]
                else:
                    non_terminal_end = fallback(non_terminal_end)
                    break

        td = defaultdict(list)
        for src, dst in enumerate(space_layer):
            td[dst].append(src)
        td = sorted(td.items(), key = lambda pc: pc[0])
        add_weight = add_weight_base and lid < len(layers_of_weight)
        
        combined = []
        new_track_nodes = []
        track_count = len(track_nodes)
        for pid, cids in td:
            if len(cids) > 1:
                if add_weight:
                    for cid in cids:
                        track_nid = track_nodes[cid]
                        non_terminals[non_terminal_end] = f'{layers_of_weight[lid][cid, 0] * 100:.0f}%'
                        top_down[non_terminal_end].add(track_nid)
                        track_nodes[cid] = non_terminal_end
                        weight_nodes[track_nid] = lid
                        non_terminal_end += 1
                labels = layers_of_label[lid + 1][pid]
                labels = [labels] if perserve_sub or labels[0] in '#_' else labels.split('+')
                non_terminals[non_terminal_end] = labels.pop()
                # >j<
                for cid in cids:
                    _combine(NTS, non_terminal_end, track_nodes[cid], non_terminals, top_down, perserve_sub)
                while labels: # unary DIY
                    non_terminal_end += 1
                    non_terminals[non_terminal_end] = labels.pop()
                    top_down[non_terminal_end] = set({non_terminal_end - 1})
                new_track_nodes.append(non_terminal_end)
                non_terminal_end += 1
                combined.extend(cids)
            else:
                new_track_nodes.append(track_nodes[cids.pop()])

        # if isinstance(fall_back_root_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and new_track_nodes == track_nodes and track_fall_back: # no action is taken
            error_layer_id = lid, E_CMB, bottom_len
            non_terminal_end = fallback(non_terminal_end)
            break

        track_nodes = new_track_nodes

    if not error_layer_id:
        if not non_terminals:
            assert fall_back_root_label, 'should provide a fallback root label'
            top_down[non_terminal_end].update(tid for tid, _, _ in terminals)
            non_terminals[non_terminal_end] = fall_back_root_label
            non_terminal_end += 1
            error_layer_id = -1, E_LBL, bottom_len

    # import pdb; pdb.set_trace()
    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, top_down.pop(nid))

    if add_weight_base:
        return terminals, top_down, non_terminal_end - 1, error_layer_id, weight_nodes
    return terminals, top_down, non_terminal_end - 1, error_layer_id


class TreeKeeper:
    @classmethod
    def from_tiger_graph(cls, graph, *args, **kw_args):
        return cls(*_read_graph(graph), *args, **kw_args)

    @classmethod
    def from_disco_penn(cls, tree, *args, **kw_args):
        args = replace_args_kwargs(_dep_n_prefix, 1, args, 'dep', kw_args)
        return cls(*_read_dpenn(tree), *args, **kw_args)

    def __init__(self, bottom_info, top_down, root_id, v2is = None, dep = None, details = False, verbose_file = None):
        if details: print('\n'.join(draw_str_lines(bottom_info, top_down)))
        if verbose_file: verbose_file = verbose_file + ('\n'.join(draw_str_lines(bottom_info, top_down)),)
        word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down, dep = dep)
        self._gaps = gap_degree(bottom, top_down, root_id) # if details else None
        self._word = word
        if v2is is None:
            bottom_tag = [node2tag[t] for t in bottom]
            l2i = None
        else:
            w2i, t2i, l2i = v2is
            bottom_tag = [t2i(node2tag[t]) for t in bottom]
            # self._dbg = [(word[tid], node2tag[bottom[tid]]) for tid, t in enumerate(bottom_tag) if t is None]
            word = [w2i(w) for w in word]
        if dep is not None:
            if details:
                print('  '.join(n.split('_')[1]+'->'+(h.split('_')[1] if h else '*') for n, h in dep.items()))
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
                if details and not bt_head:
                    print('!:', node,' misses attachment.')
            for bt_head in media:
                dep.pop(bt_head)
            if details:
                print('  '.join(n.split('_')[1]+'->'+(h.split('_')[1] if h else '*') for n, h in dep.items()))
                if media:
                    print('Removed media nodes: ' + ', '.join(media))

        self._word_tag = word, bottom_tag
        self._materials = bottom, node2tag, bottom_unary, top_down, l2i, root_id, dep, verbose_file
        self._balanced_top_down = None
        # self._dep_signals = None

    @property
    def has_signals(self):
        return any(self._materials[2:4])

    @property
    def gaps(self):
        return max(self._gaps.values())

    @property
    def lines(self):
        return self._lines

    @property
    def word(self):
        # text
        return self._word

    @property
    def word_tag(self):
        return self._word_tag
    
    def stratify(self, factor = F_LEFT, balancing = False):
        bottom, node2tag, bottom_unary, top_down, l2i, root_id, dep, vf = self._materials
        if balancing:
            if self._balanced_top_down is None:
                self._balanced_top_down = add_efficient_subs(top_down, root_id)
            top_down = self._balanced_top_down
        if factor == F_DEP:
            assert isinstance(dep, dict)
        elif isinstance(factor, dict):
            dep = factor # check static for cache?
            factor = F_DEP
        bottom = bottom if len(bottom) > 1 else bottom.copy() # pop bottom
        if factor == F_DEP:
            # if self._dep_signals is None TODO 
            return cross_signals(bottom, node2tag, bottom_unary, top_down, root_id, factor, l2i, _new_dep(dep), vf)
        return cross_signals(bottom, node2tag, bottom_unary, top_down, root_id, factor, l2i)

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


from data.mp import DM
from data.cross.evalb_lcfrs import export_string

class MxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, segments, *data_gen):
        for seg_length, word, tag, label, space in zip(*data_gen):
            layers_of_label = []
            layers_of_space = []
            label_start = 0
            for l_size, l_len in zip(segments, seg_length):
                label_end = label_start + l_len
                label_layer = label[label_start: label_end]
                layers_of_label.append(tuple(i2l(i) for i in label_layer))
                if l_len == 1:
                    break
                layers_of_space.append(space[label_start: label_end])
                label_start += l_size
            ln = seg_length[0]
            wd = [i2w[i] for i in word[:ln]]
            tg = [i2t[i] for i in  tag[:ln]]
            bt, td, rt, _ = disco_tree(wd, tg, layers_of_label, layers_of_space, 'VROOT')
            yield export_string(bid_offset, bt, td, rt)
            bid_offset += 1

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, t_args):
        bid_offset, segments = t_args[:2]
        start = seg_id * seg_size
        if start < batch_size:
            return (bid_offset + start, segments) + tuple(x[start: (seg_id + 1) * seg_size] for x in t_args[2:])