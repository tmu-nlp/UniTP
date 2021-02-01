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

import unicodedata
def len_ea(string, ea_unit = 2):
    nfc_string = unicodedata.normalize('NFC', string)
    ea_len = sum((ea_unit if unicodedata.east_asian_width(c) in 'WF' else 1) for c in nfc_string)
    return int(ea_len + 0.5)