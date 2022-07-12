from collections import namedtuple, defaultdict
from utils.str_ops import count_wide_east_asian
from utils.math_ops import bit_span, n_bit, low_bit
Span = namedtuple('Span', 'label, strokes, cur, mai')
Element = namedtuple('Element', 'bits, cur, mai')
Stroke = namedtuple('Stroke', 'cur, mai, ftag')
draw_stroke = lambda s_char, length, e_char: (s_char * length, e_char)
Symbol = namedtuple('Symbol', 'bar, left, middle, right, combine, cross')

mai = lambda mai, n: (len(n)>1)+1 if mai else 0
def dynamic_stroke(parents, elem, cursors, pid, nid, jobs, future_jobs, bottom_up):
    ftag = parents.pop(pid)
    is_still_ma = len(parents) > 1
    cur = elem.cur
    if parents:
        cursors[cur] = is_still_ma
    else:
        assert not cursors.pop(cur), 'Child should not be MA'

    for parent in parents:
        elem = jobs[parent][nid]
        jobs[parent][nid] = future_jobs[parent][nid] = \
            Element(elem.bits, elem.cur, mai(is_still_ma, bottom_up[parent]))

    if elem.mai and is_still_ma:
        return Stroke(elem.cur, 2, ftag)
    else:
        return Stroke(elem.cur, elem.mai, ftag)

def style_1(reverse):
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
    S_CROSS  = '┼' # avoid '╋' by cur
    return Symbol(D_BAR, T_LEFT, T_MIDDLE, T_RIGHT, D_COMB, S_CROSS), dynamic_stroke

def static_stroke(parents, elem, cursors, pid, *_):
    ftag = parents.pop(pid)
    if not parents:
        cursors.pop(elem.cur)
    if elem.mai and parents:
        return Stroke(elem.cur, 2, ftag)
    else:
        return Stroke(elem.cur, elem.mai, ftag)
    
def style_2(reverse):
    D_BAR = '│║'
    if reverse:
        T_LEFT   = '┌╓╟'
        T_MIDDLE = '┬╥╫'
        T_RIGHT  = '┐╖╢'
        D_COMB   = '┴╨'
    else:
        T_LEFT   = '└╙╟'
        T_MIDDLE = '┴╨╫'
        T_RIGHT  = '┘╜╢'
        D_COMB   = '┬╥'
    S_CROSS  = '┼'
    return Symbol(D_BAR, T_LEFT, T_MIDDLE, T_RIGHT, D_COMB, S_CROSS), static_stroke

def lcenter(text, num):
    if num - len(text) < 2:
        return text.ljust(num)
    return text.center(num)

def draw_bottom(bottom, top_down, wrap_len):
    bottom_up = defaultdict(dict)
    for pid, td in top_down.items():
        for cid in td.children:
            bottom_up[cid][pid] = td.children[cid]
    word_line = ''
    tag_line  = ''
    cursors = {}
    jobs = defaultdict(dict)
    for eid, (bid, word, tag) in enumerate(bottom):
        num_wide_chars = count_wide_east_asian(word)
        unit_len = max(len(word) + num_wide_chars, len(tag)) + wrap_len
        word_line += lcenter(word, unit_len - num_wide_chars)
        tag_line  += lcenter(tag,  unit_len)
        bit    = 1 << eid
        cursor = len(tag_line) - round(unit_len // 2)
        cursors[cursor] = is_ma = len(parents := bottom_up[bid]) > 1
        for parent in parents:
            jobs[parent][bid] = Element(bit, cursor, is_ma)
    return ''.join(word_line), ''.join(tag_line), bottom_up, jobs, cursors


def __dir_arity(elements, from_left, reverse):
    bits = 0
    for elem in elements.values():
        bits |= elem.bits
    lhs = n_bit(low_bit(bits))
    rhs = n_bit(bits)
    key = lhs if from_left else -rhs
    return (len(elements), rhs - lhs, key) if reverse else (key, rhs - lhs, len(elements))

sort_by_arity = lambda jobs: sorted(jobs.items(), key = lambda x: len(x[1]))
sort_by_lhs_arity = lambda jobs: sorted(jobs.items(), key = lambda x: __dir_arity(x[1], True, False))
sort_by_rhs_arity = lambda jobs: sorted(jobs.items(), key = lambda x: __dir_arity(x[1], False, False))
sort_by_arity_lhs = lambda jobs: sorted(jobs.items(), key = lambda x: __dir_arity(x[1], True, True))
sort_by_arity_rhs = lambda jobs: sorted(jobs.items(), key = lambda x: __dir_arity(x[1], False, True))

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

def replace_char(line, bars, cursors, targets = ' ─'):
    new_line = []
    last_cur = 0
    for cur, is_ma in sorted(cursors.items()):
        if line[cur] in targets:
            new_line.append(line[last_cur:cur])
            new_line.append(bars[is_ma])
            last_cur = cur + 1
    if (cur := len(line)) not in cursors:
        new_line.append(line[last_cur:])
    return ''.join(new_line)

label_only = lambda n, td: td[n].label
brace_ftag = lambda t: '{' + t + '}'
sbrkt_ftag = lambda t: '[' + t + ']'
__get_cur = lambda elem_or_span: elem_or_span.cur
from utils.param_ops import get_sole_key
def make_spans(bottom_up, top_down, jobs, cursors, label_fn, sort_jobs, stroke_fn, has_ma):
    spans = []
    span_ranges = set() if has_ma else 0
    new_cursors = {}
    future_jobs = defaultdict(dict)
    for pid, elements in sort_jobs(jobs):
        if (n := len(elements)) < len(top_down[pid].children) or \
            n == 1 < len(bottom_up[get_sole_key(elements)]):
            future_jobs[pid].update(elements)
            continue

        bits = span_bits = 0
        lhs = rhs = None
        for elem in elements.values():
            bits |= elem.bits
            if lhs is None or elem.cur < lhs: lhs = elem.cur
            if rhs is None or elem.cur > rhs: rhs = elem.cur
        
        span_bits |= bit_span(bits)
        if has_ma:
            overlapping = any((rhs-l)*(lhs-r)<=0 for l,r in span_ranges)
        else:
            overlapping = span_bits & span_ranges
        if overlapping:
            future_jobs[pid].update(elements)
            continue
        if has_ma:
            span_ranges.add((lhs, rhs))
        else:
            span_ranges |= span_bits

        cursor = 0
        strokes_for_span = []
        for nid, elem in elements.items():
            cursor += elem.cur
            strokes_for_span.append(stroke_fn(bottom_up[nid], elem, cursors, pid, nid, jobs, future_jobs, bottom_up))

        label = label_fn(pid, top_down)
        if n > 1:
            strokes_for_span.sort(key = __get_cur)
            cursor //= n
            span_curs = []
            for stroke in strokes_for_span:
                if not stroke.mai and stroke.cur == cursor:
                    span_curs = None; break
                span_curs.append(stroke.cur)
            if span_curs:
                r_len = len(label); l_len = r_len >> 1; r_len -= l_len
                if any(cur + l_len > cursor or cursor < cur - r_len for cur in span_curs):
                    cursor = dodge_cursor(span_curs, cursors)

        span_parents = bottom_up[pid]
        span_is_ma = len(span_parents) > 1
        new_cursors[cursor] = span_is_ma
        spans.append(Span(label, strokes_for_span, cursor, span_is_ma))
        for parent in span_parents:
            future_jobs[parent][pid] = Element(bits, cursor, span_is_ma)

    add_bar = cursors.copy()
    cursors.update(new_cursors)
    spans.sort(key = __get_cur) # sort if len(es) == 1
    return spans, add_bar, future_jobs

def draw_label(cursor, cur, label):
    f_len = len(label)
    num_char = cur - cursor - (f_len >> 1)
    return draw_stroke(' ', num_char, label), num_char + f_len

def draw_line(l2r_non_overlapping_spans, width, symbol, add_bar, add_bar_for_ftag, ftag_fn):
    line_ftag = []
    line_line = []
    line_cons = []
    cursor_ftag = cursor_line = cursor_cons = 0
    for span in l2r_non_overlapping_spans:
        if len(ss := span.strokes) == 1: # unary
            stroke = ss[0];
            num_char = stroke.cur - cursor_line
            line_line.extend(draw_stroke(' ', num_char, symbol.bar[stroke.mai]))
            cursor_line += num_char + 1
        else:
            num_elem = len(ss)
            for eid, stroke in enumerate(ss):
                stroke_cur = stroke.cur
                num_char = stroke_cur - cursor_line
                cursor_end = cursor_line + num_char # avoid cascade ':='s for the conditional updates from the second 
                if eid == 0:
                    line_line.extend(draw_stroke(' ', num_char, symbol.left[stroke.mai]))
                elif cursor_line < (cur := span.cur) < cursor_end: # with parent in middel
                    pre_num_char = cur - cursor_line
                    line_line.extend(draw_stroke('─', pre_num_char, symbol.combine[span.mai]))
                    sign = (symbol.middle, symbol.right)[eid == num_elem - 1][stroke.mai]
                    line_line.extend(draw_stroke('─', num_char - pre_num_char - 1, sign))
                else:
                    if eid == num_elem - 1:
                        sign = symbol.right[stroke.mai]
                    elif cur == cursor_end:
                        sign = symbol.cross
                    else:
                        sign = symbol.middle[stroke.mai]
                    line_line.extend(draw_stroke('─', num_char, sign))
                cursor_line += num_char + 1

        cons, num_char = draw_label(cursor_cons, span.cur, span.label)
        line_cons.extend(cons)
        cursor_cons += num_char

        if add_bar_for_ftag:
            for stroke in (s for s in ss if s.ftag):
                ftag, num_char = draw_label(cursor_ftag, stroke.cur, ftag_fn(stroke.ftag))
                line_ftag.extend(ftag)
                cursor_ftag += num_char

    line_line = ''.join(line_line) + ' ' * (width - cursor_line)
    line_cons = ''.join(line_cons) + ' ' * (width - cursor_cons)
    line_line = replace_char(line_line, symbol.bar, add_bar)
    line_cons = replace_char(line_cons, symbol.bar, add_bar)
    if cursor_ftag and add_bar_for_ftag:
        line_ftag = ''.join(line_ftag) + ' ' * (width - cursor_ftag)
        line_ftag = replace_char(line_ftag, symbol.bar, add_bar_for_ftag)
        return line_ftag, line_line, line_cons
    return line_line, line_cons

def ruler(width, reverse = True):
    height = len(str(width)) 
    lines = [[] for _ in range(height)]
    for x in range(width + 1):
        for lid in range(height):
            lines[lid].append(x // 10 ** lid % 10)
    if reverse: lines.reverse()
    return '\n'.join(''.join(str(x) for x in line) for line in lines)