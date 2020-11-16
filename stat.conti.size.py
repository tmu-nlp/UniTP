from data.delta import xtype_to_logits
from data.trapezoid.dataset import distribute_jobs, Queue, PennWorker, PennTreeKeeper, mp_workers, Tree
from data.penn_types import CorpusReader
from nltk.corpus import BracketParseCorpusReader
from data.io import load_i2vs, encapsulate_vocabs
from utils.types import num_threads, E_ORIF4
from utils.str_ops import strange_to

base_name = '001'

def from_penn(vocab_path, corp_path, trapezoid_height, factors):

    i2vs = load_i2vs(vocab_path, 'word tag label'.split())
    i2vs, field_v2is = encapsulate_vocabs(i2vs, {})

    _, w2i = field_v2is['word']
    _, t2i = field_v2is['tag']
    _, l2i = field_v2is['label']
    x2i = lambda x: xtype_to_logits(x, to_str = False)
    v2is = w2i, t2i, l2i, x2i
    
    if vocab_path.endswith('ptb'):
        reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")
    elif vocab_path.endswith('ktb'):
        reader = CorpusReader(corp_path)
        reader.break_corpus(400)
    else:
        reader = CorpusReader(corp_path)
    fnames = reader.fileids()
    if vocab_path.endswith('ctb'):
        data_split = strange_to('001-325,440-1151', lambda x: f'{x:04}')
        fnames = [fn for fn in reader.fileids() if fn[5:-3] in data_split]
    works = distribute_jobs(fnames, num_threads)
    q = Queue()
    for i in range(num_threads):
        w = PennWorker(q, reader, works[i], trapezoid_height, v2is, factors)
        w.start()
        works[i] = w
    def core_fn(words, length, tree_str, factored):
        keeper = PennTreeKeeper(Tree.fromstring(tree_str), v2is, trapezoid_height)
        keeper.update_factored(factored, words)
        return words, length, keeper
    text, lengths, keepers = mp_workers(works, q, core_fn, num_threads)
    return keepers

from collections import defaultdict
def to_csv(keepers, corp_name, factors):
    ratios = {fct: defaultdict(int) for fct in factors}
    with open(f'R_ggplot/parse_{corp_name}.csv', 'w') as fw:
        fw.write('len,' + ','.join(factors) + '\n')
        for keeper in keepers:
            line = []
            for fct in factors:
                token_size = len(keeper[fct]['token'])
                label_size = 0
                for x in keeper[fct]['label']:
                    this_size = len(x)
                    if label_size:
                        ratios[fct][(last_size, this_size)] += 1
                    label_size += this_size
                    last_size = this_size
                line.append(label_size)
            line = [token_size] + line
            line = ','.join(str(x) for x in line)
            fw.write(line + '\n')
    for fct, lt_cnt in ratios.items():
        with open(f'R_ggplot/parse_{corp_name}_{fct}.csv', 'w') as fw:
            fw.write('size,ratio,count\n')
            for (last_size, this_size), cnt in lt_cnt.items():
                fw.write(f'{last_size},{this_size/last_size},{cnt}\n')

    with open(f'R_ggplot/orient_{corp_name}.csv', 'w') as fw:
        fw.write('fct,lft,non,rgh\n')
        for fct in E_ORIF4:
            nlr = defaultdict(int)
            with open(join(base_path, 'data', corp_name, f'stat.xtype.{fct}')) as fr:
                for line in fr:
                    xtype, cnt = line.split()
                    nlr[xtype[0]] += int(cnt)
            fw.write(f"{fct},{nlr['<']},{nlr['-']},{nlr['>']}\n")

data = from_penn('001/data/ctb', '/cldata/LDC/ctb9.0/data/bracketed', 1, E_ORIF4)
to_csv(data, 'ctb', E_ORIF4)

data = from_penn('001/data/ktb', '/cldata/treebanks/KeyakiTreebank/treebank/', 1, E_ORIF4)
to_csv(data, 'ktb', E_ORIF4)

data = from_penn('001/data/ptb', '/cldata/LDC/penn_treebank_3/treebank_3/parsed/mrg/wsj', 1, E_ORIF4)
to_csv(data, 'ptb', E_ORIF4)