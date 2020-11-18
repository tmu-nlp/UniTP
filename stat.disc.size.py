from utils.types import M_TRAIN, M_DEVEL, M_TEST, E_ORIF5_HEAD, E_ORIF4
from sys import argv
from collections import defaultdict
from os.path import join
from tqdm import tqdm

_, base_path = argv

def to_csv(corp_name, factors):
    ratios = {fct: defaultdict(int) for fct in factors}
    offset = 0
    lines = defaultdict(list)

    for mode in (M_TRAIN, M_DEVEL, M_TEST):
        for factor in factors:
            with open(join(base_path, 'data', corp_name, f'{mode}.index.{factor}')) as fr:
                for lid, line in tqdm(enumerate(fr), desc = f'{corp_name}.{factor}'):
                    if line == '\n':
                        finish = True
                        continue
                    lengths = tuple(int(x) for x in line.split())
                    for sizes in zip(lengths, lengths[1:]):
                        ratios[factor][sizes] += 1

                    label_size = str(sum(lengths))
                    if factor == factors[0]:
                        lines[offset + lid].extend([str(lengths[0]), label_size])
                    else:
                        lines[offset + lid].append(label_size)
                assert finish

    with open(f'R_ggplot/parse_{corp_name}.csv', 'w') as fw:
        fw.write('len,' + ','.join(factors) + '\n')
        for lengths in lines.values():
            if len(lengths) == len(factors) + 1:
                fw.write(','.join(lengths) + '\n')
                
    for fct, lt_cnt in ratios.items():
        with open(f'R_ggplot/parse_{corp_name}_{fct}.csv', 'w') as fw:
            fw.write('size,ratio,count\n')
            for (last_size, this_size), cnt in lt_cnt.items():
                fw.write(f'{last_size},{this_size/last_size},{cnt}\n')
    
    with open(f'R_ggplot/orient_{corp_name}.csv', 'w') as fw:
        fw.write('fct,lft,non,rgh\n')
        for fct in factors:
            nlr = defaultdict(int)
            with open(join(base_path, 'data', corp_name, f'stat.xtype.{fct}')) as fr:
                for line in fr:
                    xtype, cnt = line.split()
                    nlr[xtype[0]] += int(cnt)
            fw.write(f"{fct},{nlr['<']},{nlr['-']},{nlr['>']}\n")

to_csv('tiger', E_ORIF5_HEAD)
to_csv('dptb', E_ORIF5_HEAD)

# to_csv('ptb')
# to_csv('ctb')
# to_csv('ktb')