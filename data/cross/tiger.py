from data.cross import TopDown, C_VROOT
from utils.param_ops import get_sole_key
get_tiger_label = lambda x: None if x == '--' else x
get_tiger_inode = lambda x: x[x.rindex('_') + 1:]
C_TIGER_NT_START = 500
get_inode = lambda x: x if x < C_TIGER_NT_START else (499 - x) # T&NT

def read(graph, make_up_for_no_nt = True):
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

    if not top_down and make_up_for_no_nt:
        top_down[0] = TopDown(C_VROOT, {bid: None for bid, _, _ in bottom})
    if 0 not in top_down:
        children = set()
        for td in top_down.values():
            children.update(td.children.keys()) # for future multi-attachment
        top_down[0] = top_down.pop(get_sole_key(top_down.keys() - children))
    return bottom, top_down