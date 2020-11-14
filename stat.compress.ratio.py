from os import listdir
from math import sqrt

def avg_fn(fname):
    situations = []
    with open(fname) as fr:
        total_count = 0
        total_ratio = 0
        for lid, line in enumerate(fr):
            if lid:
                size, ratio, count = line.split(',')
                size = int(size)
                # if size < 40: # size < start:# or size > end:
                #     continue
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

files = {}
for x in listdir('R_ggplot'):
    if x.startswith('parse_'):
        y = x.split('_')
        if len(y) == 3:
            _, corp_name, factor = y
            factor = factor[:-4]
            files[(corp_name, factor)] = avg_fn(f'R_ggplot/{x}')

for corp_name in ('ptb', 'ctb', 'ktb'):
    for factor in ('left', 'right', 'midin', 'midout'):
        avg, std, cnt = files[(corp_name, factor)]
        cnt = f'{cnt / 10 ** 6:.2f}M'
        print(f'${avg:.2f}\\ _{{\pm{std:.2f}}}$', end = ' & ')
    print()

print()
for corp_name in ('dptb', 'tiger'):
    for factor in reversed(('left', 'midin25', 'midin', 'midin75', 'right')): # TODO: reverse
        avg, std, cnt = files[(corp_name, factor)]
        cnt = f'{cnt / 10 ** 6:.2f}M'
        print(f'${avg:.2f}\\ _{{\pm{std:.2f}}}$', end = ' & ')
    print()