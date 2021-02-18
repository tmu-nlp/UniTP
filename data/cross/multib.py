from data.cross import has_multiple, TopDown, _read_dpenn, _read_graph, _pre_proc
from random import random

def _closest(order, children):
    if isinstance(order, float):
        order = round(order * (len(children) - 1e-10) - 0.5)
    return children[order]

def _multi_hash(bottom,
                top_down,
                completed_nodes,
                factor):
    # make a new td: including head flag
    bottom_flag = [True for _ in bottom]
    bottom_trace = [None for _ in bottom]
    p_head = {}
    for p_node, td in top_down.items():
        if p_node in completed_nodes or not all(node in bottom for node in td.children):
            continue

        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location_children.append((nid, node))
                bottom_trace[nid] = node

        if factor is False:
            factor = random()
        if isinstance(factor, (int, float)):
            p_head[_closest(factor, location_children)[1]] = p_node, [nid for nid, _ in location_children]
    return bottom_trace, p_head


def cross_signals(bottom, node2tag, bottom_unary, top_down, factor, # float:midin True:most_continuous False:random None:dep
                  dependency = None,
                  pos_prefix = '#'):
    # sub_suffix = '.'

    layers_of_label = []
    layers_of_disco = []
    completed_nodes = set()
    while len(bottom) > 1:
        label_layer = []
        for node in bottom:
            if node in bottom_unary:
                label_layer.append(bottom_unary[node])
            elif node in node2tag:
                label_layer.append(pos_prefix + node2tag[node])
            else:
                label_layer.append(top_down[node].label)

        bottom_trace, p_head = _multi_hash(bottom, top_down, completed_nodes, factor)
        new_bottom = []
        disco_layer = [None for _ in bottom]
        counter = 0
        for bid, t in enumerate(bottom_trace):
            if t is None:
                new_bottom.append(bottom[bid])
                disco_layer[bid] = counter
                counter += 1
            elif t in p_head:
                p_node, children = p_head.pop(t)
                completed_nodes.add(p_node)
                new_bottom.append(p_node)
                for t in children:
                    disco_layer[t] = counter
                counter += 1

        bottom = new_bottom
        layers_of_label.append(label_layer)
        layers_of_disco.append(disco_layer)
    layers_of_label.append([top_down[bottom.pop()].label])
    assert not bottom

    return layers_of_label, layers_of_disco


from data.cross import E_SHP, E_CMB, defaultdict, _combine, draw_str_lines, gap_degree
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_disco,
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

    for lid, disco_layer in enumerate(layers_of_disco):
        td = defaultdict(list)
        for src, dst in enumerate(disco_layer):
            td[dst].append(src)
        combined = []
        new_track_nodes = []
        for pid, cids in sorted(td.items(), key = lambda pc: pc[0]):
            if len(cids) > 1:
                labels = layers_of_label[lid + 1][pid]
                labels = [labels] if perserve_sub or labels[0] in '#_' else labels.split('+')
                non_terminals[non_terminal_start] = labels.pop()
                # >j<
                for cid in cids:
                    _combine(NTS, non_terminal_start, track_nodes[cid], non_terminals, top_down, perserve_sub)
                while labels: # unary
                    non_terminal_start += 1
                    non_terminals[non_terminal_start] = labels.pop()
                    top_down[non_terminal_start] = set({non_terminal_start - 1})
                new_track_nodes.append(non_terminal_start)
                non_terminal_start += 1
                combined.extend(cids)
            else:
                new_track_nodes.append(track_nodes[cids.pop()])

        # if isinstance(fall_back_root_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and new_track_nodes == track_nodes and track_fall_back: # no action is taken
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

        track_nodes = new_track_nodes

    # import pdb; pdb.set_trace()
    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, top_down.pop(nid))

    return terminals, top_down, non_terminal_start - 1, error_layer_id


class TreeKeeper:
    @classmethod
    def from_tiger_graph(cls, graph, *args, **kw_args):
        return cls(*_read_graph(graph), *args, **kw_args)

    @classmethod
    def from_disco_penn(cls, tree, *args, **kw_args):
        return cls(*_read_dpenn(tree), *args, **kw_args)

    def __init__(self, bottom_info, top_down, root_id, details = False):
        self._lines = draw_str_lines(bottom_info, top_down) if details else None
        word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
        self._gaps = gap_degree(bottom, top_down, root_id) if details else None
        bottom_tag = [node2tag[t] for t in bottom]
        self._word_tag = word, bottom_tag
        self._materials = bottom, node2tag, bottom_unary, top_down

    @property
    def gaps(self):
        return max(self._gaps.values())

    @property
    def lines(self):
        return self._lines

    @property
    def word_tag(self):
        return self._word_tag
    
    def stratify(self, factor):
        bottom, node2tag, bottom_unary, top_down = self._materials
        return cross_signals(bottom, node2tag, bottom_unary, top_down, factor)