
def all_paths(top_down, nid = 0):
    end_prefix = {nid: {(nid,)}}
    descendants = dict(dump_path(end_prefix, top_down, nid, nid))
    return descendants, end_prefix

def dump_path(end_prefix, top_down, nid, anti_loop):
    if nid in top_down:
        descendants = set()
        for cid in top_down[nid].children:
            if cid == anti_loop:
                continue
            cp = set(p + (cid,) for p in end_prefix[nid])
            if cid in end_prefix:
                end_prefix[cid].update(cp)
            else:
                end_prefix[cid] = cp
            for k, d in dump_path(end_prefix, top_down, cid, anti_loop):
                yield k, d
                descendants.update(d)
            descendants.add(cid)
        yield nid, descendants

class PathFinder:
    def __init__(self, top_down, root_id = 0):
        self._top_down = top_down
        self._from, self._paths = all_paths(top_down, root_id)

    def gen_from(self, pid):
        for end in self._from[pid]:
            for path in self._paths[end]:
                if pid in path:
                    yield path[path.index(pid):]

    def gen_from_to(self, pid, cid):
        for path in self._paths[cid]:
            if pid in path:
                yield path[path.index(pid):]

    def find_labeled_on_path(self, label, child_id, low = True):
        for path in self._paths[child_id]:
            if low: path = path[::-1] # necessary to match 1 DPTB sample with two PRN
            for pid, nid in zip(path[1:], path):
                if self._top_down[nid].label == label:
                    return pid, nid
        raise KeyError(f'Label \'{label}\' not found!')

    def __getitem__(self, cid):
        return self._paths[cid]


import xml.etree.ElementTree as ET

class XMLWriter:
    def __init__(self):
        self._lines = []

    def add(self, bottom, top_down, root_id = 0):
        s = ET.Element('s', id = str(len(self._lines) + 1))
        graph = ET.SubElement(s, 'graph', {}, root = str(root_id)) # TODO level in {ctree, dtree, dag}
        self._lines.append(graph)

        terminals = ET.SubElement(graph, 'terminals')
        nonterminals = ET.SubElement(graph, 'nonterminals')

        for bid, word, tag in bottom:
            ET.SubElement(terminals, 't', id = str(bid), word = word, pos = tag)
        
        for nid, td in top_down.items():
            nt = ET.SubElement(nonterminals, 'nt', id = str(nid), cat = td.label)
            for cid, ftag in td.children.items():
                ET.SubElement(nt, 'edge', label = ftag if ftag else '--', idref = str(cid))

    def dump(self, fname):
        body = ET.Element('body')
        body.extend(self._lines)
        tree = ET.ElementTree(body)
        ET.indent(tree, space = ' ' * 4, level = 0)
        tree.write(fname)