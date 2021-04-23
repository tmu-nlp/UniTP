from data.cross import has_multiple, TopDown, _read_dpenn, _read_graph, _pre_proc
from data.cross import defaultdict, gap_degree, height_gen, add_efficient_subs
from random import random

F_RANDOM = 'random'
F_LEFT = 'left'
F_RIGHT = 'right'
F_DEP = 'dep'
F_CON = 'continuous'
E_FACTOR = F_RANDOM, F_LEFT, F_RIGHT, F_DEP, F_CON

def _closest(order, length):
    if isinstance(order, float):
        order = round(order * (length - 1e-10) - 0.5)
    return order

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

        if isinstance(factor, str):
            if factor == F_RANDOM:
                factor = random()
            elif factor == F_LEFT:
                factor = 0
            elif factor == F_RIGHT:
                factor = -1
            else:
                raise NotImplementedError()
        elif isinstance(factor, dict):
            loc_in = {node: nid for nid, node in enumerate(children)}
            dep_in = defaultdict(int)
            for nid, node in enumerate(children):
                node_in = factor[node]
                if node_in in loc_in:
                    dep_in[loc_in[node_in]] += 1
            factor = max(dep_in, key = lambda x: dep_in[x])

        cid = _closest(factor, len(children))
        p_head[children[cid]] = p_node, None, location
    return bottom_trace, p_head


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
                  pos_prefix = '#'):
    # sub_suffix = '.'
    f_continuous = factor == F_CON
    if f_continuous:
        mid_in = dependency_or_mid_in
        boundaries = {}
        bottom_ref = {node: i for i, node in enumerate(bottom)}
        bottom_up = {}
        gaps = gap_degree(bottom, top_down, root_id)
    elif factor == F_DEP:
        factor = dependency_or_mid_in

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

    return terminals, top_down, non_terminal_end - 1, error_layer_id


class TreeKeeper:
    @classmethod
    def from_tiger_graph(cls, graph, *args, **kw_args):
        return cls(*_read_graph(graph), *args, **kw_args)

    @classmethod
    def from_disco_penn(cls, tree, *args, **kw_args):
        return cls(*_read_dpenn(tree), *args, **kw_args)

    def __init__(self, bottom_info, top_down, root_id, v2is = None, dep = None, details = False):
        self._lines = draw_str_lines(bottom_info, top_down) if details else None
        word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
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

        self._word_tag = word, bottom_tag
        self._materials = bottom, node2tag, bottom_unary, top_down, l2i, root_id, dep
        self._balanced_top_down = None

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
    
    def stratify(self, factor = F_RANDOM, balancing = False):
        bottom, node2tag, bottom_unary, top_down, l2i, root_id, dep = self._materials
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
            return cross_signals(bottom, node2tag, bottom_unary, top_down, root_id, factor, l2i, dep)
        return cross_signals(bottom, node2tag, bottom_unary, top_down, root_id, factor, l2i)