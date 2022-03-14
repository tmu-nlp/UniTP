from data.cross import TopDown, C_VROOT
from utils.param_ops import get_sole_key
from collections import defaultdict, namedtuple
get_tiger_label = lambda x: None if x == '--' else x
get_tiger_inode = lambda x: x[x.rindex('_') + 1:]
C_TIGER_NT_START = 500
get_inode = lambda x: x if x < C_TIGER_NT_START else (499 - x) # T&NT

def make_root(bottom, top_down, make_up_for_no_nt):
    if not top_down and make_up_for_no_nt:
        top_down[0] = TopDown(C_VROOT, {bid: None for bid, _, _ in bottom})
    if 0 not in top_down:
        parents = set(top_down.keys())
        for td in top_down.values():
            parents -= td.children.keys()
        top_down[0] = top_down.pop(get_sole_key(parents))

def read_tree(graph, make_up_for_no_nt = True):
    top_down = {}
    single_attachment = set()
    terminals, non_terminals = graph[0]
    bottom = [(int(get_tiger_inode(t.get('id'))), t.get('word'), t.get('pos')) for t in terminals]

    for nt in non_terminals:
        p_node = get_tiger_inode(nt.get('id'))
        p_node = get_inode(int(p_node)) if p_node.isdigit() else 0
        label = nt.get('cat')
        children = {}

        for edge in nt:
            if edge.tag == 'secedge':
                continue
            node = get_inode(int(get_tiger_inode(edge.get('idref'))))
            children[node] = get_tiger_label(edge.get('label'))
            assert node not in single_attachment, 'multi-attachment'
            single_attachment.add(node)
            top_down[p_node] = TopDown(label, children)

    make_root(bottom, top_down, make_up_for_no_nt)
    return bottom, top_down

MultiAttach = namedtuple('MultiAttach', 'pid, label')
def read_graph(graph, make_up_for_no_nt = True):
    top_down = {}
    multi_attachment = defaultdict(list)
    terminals, non_terminals = graph[0]
    bottom = []
    for t in terminals:
        tid = int(get_tiger_inode(t.get('id')))
        bottom.append((tid, t.get('word'), t.get('pos')))
        t_sub = (t[i] for i,s in enumerate(t) if s.tag == 'secedge')
        for s in t_sub:
            idref = get_inode(int(get_tiger_inode(s.get('idref'))))
            multi_attachment[tid].append(MultiAttach(idref, s.get('label')))

    for nt in non_terminals:
        p_node = get_tiger_inode(nt.get('id'))
        p_node = get_inode(int(p_node)) if p_node.isdigit() else 0
        label = nt.get('cat')
        children = {}

        for edge in nt:
            idref = get_inode(int(get_tiger_inode(edge.get('idref'))))
            f_tag = get_tiger_label(edge.get('label'))
            if edge.tag == 'secedge':
                if idref < 0:
                    multi_attachment[p_node].append(MultiAttach(idref, f_tag))
                else:
                    assert label is None, f'{p_node} {idref}'
                continue
            else: # e+s e+m
                assert idref not in children
                children[idref] = label
            top_down[p_node] = TopDown(label, children)

    for node, parents in multi_attachment.items():
        for p_node, label in parents:
            children = top_down[p_node].children
            assert node not in children
            children[node] = label

    make_root(bottom, top_down, make_up_for_no_nt)
    return bottom, top_down