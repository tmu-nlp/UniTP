from collections import namedtuple, defaultdict
from utils.str_ops import count_wide_east_asian
Span = namedtuple('Span', 'label, elements, cur, mai')
Element = namedtuple('Element', 'bits, cur, mai')
draw_elem = lambda s_char, length, e_char: (s_char * length, e_char)

def symbols(reverse):
    S_CROSS  = '┼' # avoid '╋' by cur
    D_BAR = '│┃'
    if reverse:
        T_LEFT   = '┌┟┠'
        T_MIDDLE = '┬╁╂'
        T_RIGHT  = '┐┧┨'
        D_COMB   = '┴┸'
    else:
        T_LEFT   = '└┞┠'
        T_MIDDLE = '┴╀╂'
        T_RIGHT  = '┘┦┨'
        D_COMB   = '┬┰'
    return D_BAR, T_LEFT, T_MIDDLE, T_RIGHT, D_COMB, S_CROSS

mai = lambda mai, n: (len(n)>1)+1 if mai else 0
def draw_bottom(bottom, top_down, wrap_len):
    bottom_up = defaultdict(set)
    for pid, td in top_down.items():
        for cid in td.children:
            bottom_up[cid].add(pid)
    word_line = ''
    tag_line  = ''
    cursors = {}
    jobs = defaultdict(dict)
    wl = wrap_len << 1
    for eid, (bid, word, tag) in enumerate(bottom):
        num_wide_chars = count_wide_east_asian(word)
        unit_len = max(len(word) + num_wide_chars, len(tag)) + wl
        word_line += word.center(unit_len - num_wide_chars)
        tag_line  +=  tag.center(unit_len)
        bit    = 1 << eid
        cursor = len(tag_line) - round(unit_len // 2)
        cursors[cursor] = is_ma = len(parents := bottom_up[bid]) > 1
        for parent in parents:
            jobs[parent][bid] = Element(bit, cursor, mai(is_ma, bottom_up[parent]))
    return ''.join(word_line), ''.join(tag_line), bottom_up, jobs, cursors

sort_by_num_children = lambda jobs: sorted(jobs.items(), key = lambda x: len(x[1]))
get_cur = lambda elem_or_span: elem_or_span.cur
update_ma = lambda elem, ma: Element(*elem[:-1], ma)

def dodge_cursor(span_curs, cursors):
    distance = 0
    cursors = set(cur for cur in cursors if span_curs[0] < cur < span_curs[-1])
    for lhs, rhs in zip(span_curs, span_curs[1:]):
        curs = sorted(cur for cur in cursors if lhs < cur < rhs)
        curs.insert(0, lhs); curs.append(rhs)
        for c_lhs, c_rhs in zip(curs, curs[1:]):
            if (dist := c_rhs - c_lhs) > distance:
                max_cursor = (c_lhs + c_rhs) >> 1
                distance = dist
    return max_cursor

def replace_char(line, D_BAR, cursors, targets):
    new_line = []
    last_cur = 0
    for cur, is_ma in sorted(cursors.items()):
        if line[cur] in targets:
            new_line.append(line[last_cur:cur])
            new_line.append(D_BAR[is_ma])
            last_cur = cur + 1
    if (cur := len(line)) not in cursors:
        new_line.append(line[last_cur:])
    return ''.join(new_line)

from utils.math_ops import bit_span
label_only = lambda n, td: td[n].label
def make_spans(bottom_up, top_down, jobs, cursors, label_fn, sort_jobs):
    future_jobs = defaultdict(dict)
    spans = []
    layer_span_bits = 0
    new_cursors = {}
    for pid, elements in sort_jobs(jobs):
        if (n := len(elements)) < len(top_down[pid].children):
            future_jobs[pid].update(elements)
            continue

        bits = span_bits = 0
        for elem in elements.values():
            bits |= elem.bits
        span_bits |= bit_span(bits)
        # allow multi_attach sharing boundary? combine neibours in to a big span
        if span_bits & layer_span_bits:
            future_jobs[pid].update(elements)
            continue

        cursor = 0
        layer_span_bits |= span_bits
        label = label_fn(pid, top_down)
        elements_for_span = []
        for nid, elem in elements.items():
            elements_for_span.append(elem)
            cursor += (cur := elem.cur)
            other_parents = bottom_up[nid]
            other_parents.remove(pid)
            is_ma = len(other_parents) > 1
            if other_parents:
                cursors[cur] = is_ma
            else:
                assert not cursors.pop(cur), 'Child should not be MA'
            for parent in other_parents:
                jobs[parent][nid] = future_jobs[parent][nid] = \
                update_ma(jobs[parent][nid], mai(is_ma, bottom_up[parent]))

        if n == 1:
            elements_for_span = elem
        else:
            elements_for_span.sort(key = get_cur)
            cursor //= n
            span_curs = []
            for elem in elements_for_span:
                if not elem.mai and elem.cur == cursor:
                    span_curs = None; break
                span_curs.append(elem.cur)
            if span_curs:
                r_len = len(label); l_len = r_len >> 1; r_len -= l_len
                if any(cur + l_len > cursor or cursor < cur - r_len for cur in span_curs):
                    cursor = dodge_cursor(span_curs, cursors)

        span_parents = bottom_up[pid]
        span_is_ma = len(span_parents) > 1
        new_cursors[cursor] = span_is_ma
        spans.append(Span(label, elements_for_span, cursor, span_is_ma))
        for parent in span_parents:
            future_jobs[parent][pid] = Element(bits, cursor, span_is_ma)

    add_bar = lambda line, D_BAR: replace_char(line, D_BAR, cursors.copy(), ' ─')
    cursors.update(new_cursors)
    spans.sort(key = get_cur) # sort if len(es) == 1
    return spans, add_bar, future_jobs

def draw_line(l2r_non_overlapping_spans, add_bar, width, D_BAR, T_LEFT, T_MIDDLE, T_RIGHT, D_COMB, S_CROSS):
    line_line = []
    line_cons = []
    cursor_line = cursor_cons = 0
    for span in l2r_non_overlapping_spans:
        if isinstance(es := span.elements, Element): # unary
            num_char = es.cur - cursor_line
            line_line.extend(draw_elem(' ', num_char, D_BAR[es.mai]))
            cursor_line += num_char + 1
        else:
            num_elem = len(es)
            for eid, elem in enumerate(es):
                elem_cur = elem.cur
                num_char = elem_cur - cursor_line
                if eid == 0:
                    line_line.extend(draw_elem(' ', num_char, T_LEFT[elem.mai]))
                elif cursor_line < (cur := span.cur) < (cursor_end := cursor_line + num_char): # with parent in middel
                    pre_num_char = cur - cursor_line
                    line_line.extend(draw_elem('─', pre_num_char, D_COMB[span.mai]))
                    sign = (T_MIDDLE, T_RIGHT)[eid == num_elem - 1][elem.mai]
                    line_line.extend(draw_elem('─', num_char - pre_num_char - 1, sign))
                else:
                    if eid == num_elem - 1:
                        sign = T_RIGHT[elem.mai]
                    elif cur == cursor_end:
                        sign = S_CROSS
                    else:
                        sign = T_MIDDLE[elem.mai]
                    line_line.extend(draw_elem('─', num_char, sign))
                cursor_line += num_char + 1

        label = span.label
        f_len = len(label)
        num_char = span.cur - cursor_cons - (f_len >> 1)
        line_cons.extend(draw_elem(' ', num_char, label))
        cursor_cons += num_char + f_len

    line_line = ''.join(line_line) + ' ' * (width - cursor_line)
    line_cons = ''.join(line_cons) + ' ' * (width - cursor_cons)
    if callable(add_bar):
        line_line = add_bar(line_line, D_BAR)
        line_cons = add_bar(line_cons, D_BAR)
    return line_line, line_cons

def ruler(width, reverse = True):
    height = len(str(width)) 
    lines = [[] for _ in range(height)]
    for x in range(width + 1):
        for lid in range(height):
            lines[lid].append(x // 10 ** lid % 10)
    if reverse: lines.reverse()
    return '\n'.join(''.join(str(x) for x in line) for line in lines)