from collections import defaultdict

def histo_count(cnt, num_bin = None, bin_size = None, sep = '\n'):
    if bin_size is None:
        max_cnt = max(cnt.keys())
        bin_size = max_cnt / num_bin

    info = []
    stat = defaultdict(int)
    total = 0
    if isinstance(bin_size, int):
        fmt1 = '%d ~ %d: '
    else:
        fmt1 = '%.2f ~ %.2f'
    for k,v in cnt.items():
        stat[k//bin_size] += v
        total += v
    for k in sorted(stat.keys()):
        s1 = fmt1 % (k*bin_size, (k+1)*bin_size)
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