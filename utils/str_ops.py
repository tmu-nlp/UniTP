from collections import defaultdict
import re
is_numeric = re.compile(r'\d*[\.\-/:,]?\d*')

def write_ptr(base, ptr, src, overwrite, as_unit):
    if isinstance(base, list):
        if as_unit:
            src = [src]
            length = 1
        else:
            src = list(src)
            src.reverse()
            length = len(src)
        while src:
            base.insert(ptr, src.pop())
            if overwrite: # same as base[ptr] = src.pop()
                base.pop(ptr + length)
        return
    if as_unit:
        src = base.__class__((src,))
        length = 1
    else:
        length = len(src)
    if overwrite:
        suffix = src + base[ptr + length:]
    else: # insert
        suffix = src + base[ptr]
    return base[:ptr] + suffix

def delete_ptr(base, ptr, length):
    if isinstance(base, list):
        for _ in range(length):
            base.pop(ptr)
        return
    return base[:ptr] + base[ptr + length:]

def swap_ptr(base, low, high, length):
    if low > high:
        low, high = high, low
    length = min(length, high - low)
    if isinstance(base, list):
        for i_ in range(length):
            high_val = base.pop(high + i_)
            low_val  = base.pop(low  + i_)
            base.insert(low  + i_, high_val)
            base.insert(high + i_, low_val)
        return
    low_end  = low  + length
    high_end = high + length
    return base[:low] + base[high:high_end] + base[low_end:high] + base[low:low_end] + base[high_end:]

def histo_count(cnt, num_bin = None, bin_size = None, bin_start = None, sep = '\n'):
    if bin_start is None:
        bin_start = min(cnt.keys())
    if bin_size is None:
        bin_size = (max(cnt.keys()) - bin_start) / num_bin

    info = []
    stat = defaultdict(int)
    total = 0
    if isinstance(bin_size, int):
        fmt1 = '%d ~ %d: '
        minx = 1
    else:
        fmt1 = '%.2f ~ %.2f'
        minx = 0.01
    for k,v in cnt.items():
        stat[(k - bin_start)//bin_size] += v
        total += v
    for k in sorted(stat.keys()):
        s1 = fmt1 % (k*bin_size + bin_start, (k+1)*bin_size + bin_start - minx)
        s1 = s1.rjust(15)
        s2 = str_percentage(stat[k] / total)
        s2 = s2.rjust(8)
        s3 = '(%d)' % stat[k]
        s3 = s3.rjust(10)
        info.append( s1 + s2 + s3)
    return sep.join(info)

def str_percentage(x, fmt = '%.2f %%'):
    return fmt % (100 * x)

def strange_to(string, unit_op = lambda x:x, include_end_to_range = True):
    # '000-002,007': ('000', '001', '002', '007')
    if isinstance(string, int):
        string = str(string)
    final = []
    groups = string.split(',')
    for g in groups:
        if '-' in g:
            start, end = (int(i) for i in g.split('-'))
            assert end > start
            if include_end_to_range:
                end += 1
            final.extend(unit_op(i) for i in range(start, end))
        elif g:
            g = int(g)
            final.append(unit_op(g))
    return final

def hex_ratio(x):
    assert 0 <= x <= 1
    return '0123456789abcdef'[int(16 * x - 1e-8)]

def width_ratio(x):
    assert 0 <= x <= 1
    if x == 0: return '-'
    return '▏▎▍▌▋▊▉'[int(7 * x - 1e-8)]

def height_ratio(x):
    assert 0 <= x <= 1
    if x == 0: return '-'
    return '▁▂▃▄▅▆▇▉'[int(8 * x - 1e-8)]

def space_height_ratio(x):
    assert 0 <= x <= 1
    return ' ▁▂▃▄▅▆▇▉'[int(9 * x - 1e-8)]

def str_ruler(length, upper = True, append_length = True):
    unit = '┃╵╵╵╵╿╵╵╵╵' if upper else '┃╷╷╷╷╽╷╷╷╷'
    rule = ''
    while len(rule) < length:
        rule += unit
    rule = rule[:length]
    if append_length:
        rule += str(length)
    return rule

def cat_lines(lhs_lines, rhs_lines, offset = 0, from_top = False):
    lhs_len = len(lhs_lines)
    rhs_len = len(rhs_lines)
    lines = []
    lhs_span = max(len(x) for x in lhs_lines)
    rhs_span = max(len(x) for x in rhs_lines)
    lhs_space = ' ' * lhs_span
    rhs_space = ' ' * rhs_span
    assert offset >= 0
    if not from_top:
        lhs_lines = lhs_lines[::-1]
        rhs_lines = rhs_lines[::-1]
    for lid, lhs_line in enumerate(lhs_lines):
        line = lhs_line + ' ' * (lhs_span - len(lhs_line))
        if 0 <= lid - offset < rhs_len:
            line += rhs_lines[lid - offset]
        # else:
        #     line += rhs_space
        lines.append(line)
    rhs_remain = lhs_len - offset
    while rhs_remain < rhs_len:
        if rhs_remain < 0:
            line = ''
        else:
            line = lhs_space + rhs_lines[rhs_remain]
        lines.append(line)
        rhs_remain += 1
    if not from_top:
        lines.reverse()
    return lines

import unicodedata
def count_wide_east_asian(string):
    nfc_string = unicodedata.normalize('NFC', string)
    return sum(unicodedata.east_asian_width(c) in 'WF' for c in nfc_string)

def zip_to_str(x, sep, str_fn):
    return sep.join(str_fn(i) for i in x)

def unzip_from_str(x, *fns):
    if len(fns) == 1:
        return fns[0](x)
    sep, fn = fns[:2]
    return fn(unzip_from_str(i, *fns[2:]) for i in x.split(sep))

class StringProgressBar:
    __active_lines__ = -1
    @classmethod
    def line(cls, desc, total, length = 30, char = '─', **kwargs):
        return cls((desc, length * char), **kwargs).update(total = total)

    def __init__(self, texts, sep = ' ', color = '5', finish_color = '2', error_color = '1'):
        if isinstance(texts, str):
            texts = texts,
        else:
            assert isinstance(texts, (list, tuple))
        self._sep = sep, texts, color, finish_color, error_color
        p, g, c, t = [], [], [], []
        for text in texts:
            p.append(0)
            c.append(0)
            t.append(None)
            g.append(len(text))
        self._progress = p, tuple(g), c, t
        self._cache = None
        self._file = None

    def update(self, idx = -1, num = None, *, total = None):
        p, g, c, t = self._progress
        if isinstance(idx, str):
            idx = self._sep[1].index(idx)
        if num is None:
            if isinstance(total, int):
                t[idx] = float(total)
                return self
            if t[idx] is None:
                num = -1
            else:
                c[idx] += 1
                num = c[idx] / t[idx]
        if isinstance(num, float):
            num = int(num * g[idx])
        if p[idx] != num:
            p[idx] = num
            self._cache = None
        if self._file:
            self.__update()
            self.__draw()
        return self

    def __update(self):
        line = []
        sep, texts, color, fc, ec = self._sep
        for text, p, g in zip(texts, *self._progress[:2]):
            if p < 0 or p > g:
                c = ec
            elif p == g:
                c = fc
            else:
                c = color
            seg = f'\033[3{c}m' + text[:p] + '\033[m' + text[p:]
            line.append(seg)
        self._cache = sep.join(line)

    def __draw(self):
        if offset := (StringProgressBar.__active_lines__ - self._line_id):
            print(f'\033[{offset}A\r' + self._cache, file = self._file, end = '\n' * offset)
        else:
            print('\r' + self._cache, file = self._file, end = '')

    def __str__(self):
        if self._cache is None:
            self.__update()
        return self._cache

    def __enter__(self):
        from sys import stderr
        self._file = stderr
        StringProgressBar.__active_lines__ += 1
        self._line_id = StringProgressBar.__active_lines__
        if self._line_id:
            print(file = self._file)
        self.__update()
        self.__draw()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file = None
        if StringProgressBar.__active_lines__ == 0:
            print(file = self._file)
        StringProgressBar.__active_lines__ -= 1