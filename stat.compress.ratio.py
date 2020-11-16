from os import listdir
from math import sqrt
from utils.types import E_ORIF4, E_ORIF5

def avg_fn(fname):
    situations = []
    with open(fname) as fr:
        total_count = 0
        total_ratio = 0
        for lid, line in enumerate(fr):
            if lid:
                size, ratio, count = line.split(',')
                size = int(size)
                if size < 40: # size < start:# or size > end:
                    continue
                count = int(count)
                ratio = float(ratio)
                situations.append((count, ratio))
                total_ratio += ratio * count
                total_count += count
    avg = total_ratio / total_count
    total_std = 0
    for count, ratio in situations:
        total_std += count * (ratio - avg) ** 2
    std = sqrt(total_std / (total_count - 1))
    return avg, std, total_count

def num_to_str(x):
    y = f'{x:.2f}'
    if x > 1:
        return y[:-1]
    return y[1:]

def ratio_fn(corp_name, triplet):
    lnr = {}
    with open(f'R_ggplot/orient_{corp_name}.csv') as fr:
        next(fr)
        for line in fr:
            fct, l, n, r = line.split(',')
            l, n, r = int(l), int(n), int(r)
            if triplet:
                if l > r:
                    line = '\\mathbf{' + num_to_str(l/n) + '}:1:' + num_to_str(r/n)
                else:
                    line = num_to_str(l/n) + ':1:\\mathbf{' + num_to_str(r/n) + '}'
            elif l > r:
                line = '\\mathbf{1}:' + f'{r/l:.2f}'
            else:
                line = f'{l/r:.2f}:\\mathbf{{1}}'
            lnr[fct] = line
    return lnr

files = {}
for x in listdir('R_ggplot'):
    if x.startswith('parse_'):
        y = x.split('_')
        if len(y) == 3:
            _, corp_name, factor = y
            factor = factor[:-4]
            files[(corp_name, factor)] = avg_fn(f'R_ggplot/{x}')

for corp_name in ('ptb', 'ctb', 'ktb'):
    lnr = ratio_fn(corp_name, False)
    for factor in E_ORIF4:
        avg, std, cnt = files[(corp_name, factor)]
        cnt = f'{cnt / 10 ** 6:.2f}M'
        print(f'${avg:.2f}\\ _{{\pm{std:.2f}}}$', end = ' & ')
    print()
    for factor in E_ORIF4:
        print(f'${lnr[factor]}$', end = ' & ')
    print()

print()
for corp_name in ('dptb', 'tiger'):
    lnr = ratio_fn(corp_name, True)
    for factor in E_ORIF5:
        avg, std, cnt = files[(corp_name, factor)]
        cnt = f'{cnt / 10 ** 6:.2f}M'
        print(f'${avg:.2f}\\ _{{\pm{std:.2f}}}$', end = ' & ')
    print()
    for factor in E_ORIF5:
        print(f'\\small ${lnr[factor]}$', end = ' & ')
    print()