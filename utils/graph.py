from collections import defaultdict, namedtuple
Group = namedtuple('Group', 'is_loop, chain')

def directed_graph(counter, edges, indent = 0):
    partial_order_set = defaultdict(set)
    is_loop = {}
    for end in counter:
        if end in is_loop:
            partial_order_set[end] = {k for k, v in partial_order_set.items() if end in v}
        else:
            explore(edges, end, partial_order_set, counter, is_loop, indent)
    loops = {x: partial_order_set.pop(x) for x in is_loop if is_loop[x]}
    return partial_order_set, loops

def explore(edges, anti_loop, pos, counter, is_loop, indent):
    if indent: print(' ' * indent + f'Start {anti_loop}   ', counter, is_loop)
    new_ends = set()
    new_edges = set()
    for start, end in edges:
        if end == anti_loop:
            if start in pos and end in pos[start]:
                is_loop[anti_loop] = True
                continue
            counter[anti_loop] += 1
            if start in is_loop:
                if indent: print(' ' * indent + f'  Skip exploring {start} and use its count {counter[start]}')
                counter[anti_loop] += counter[start]
            else:
                new_ends.add(start)
            pos[end].add(start)
        else:
            new_edges.add((start, end))
    for end in new_ends:
        if end not in is_loop:
            if indent: print(' ' * indent + f'  Explore {end} ', counter, is_loop)
            explore(new_edges, end, pos, counter, is_loop, (indent + 2) if indent else 0)
            counter[anti_loop] += counter[end]
            if is_loop[end]: is_loop[anti_loop] = True
            if indent: print(' ' * indent + f'  Explored {end}', counter, is_loop)
    if anti_loop not in is_loop:
        is_loop[anti_loop] = False
    if indent: print(' ' * indent + f'Finished {anti_loop}', counter, is_loop)