C_SSTB = 'sstb'

from sys import stderr
from nltk.tree import Tree
from collections import Counter, namedtuple, defaultdict
from os.path import join
from data.io import check_vocab, post_build, split_dict

sstb_split = split_dict('train.txt', 'dev.txt', 'test.txt')
build_params = {C_SSTB: sstb_split}
ft_bin = {C_SSTB: 'en'}

from data.io import make_call_fasttext
call_fasttext = make_call_fasttext(ft_bin)

def get_sstb_trees(fpath, split_files):
    corpus = {}
    for dst, src in split_files.items():
        dataset = []
        with open(join(fpath, src)) as fr:
            for line in fr:
                dataset.append(line)
        corpus[dst[:-4]] = dataset
    return corpus

VocabCounters = namedtuple('VocabCounters', 'length, word, polar, right')
build_counters = lambda: VocabCounters(Counter(), Counter(), Counter(), Counter())

from utils.types import M_TRAIN
def build(save_to_dir, fpath, corp_name, **split_files):
    assert corp_name == C_SSTB
        
    from data.mp import mp_while, Process

    class WorkerX(Process):
        estimate_total = False

        def __init__(self, *args):
            Process.__init__(self)
            self._q_jobs = args

        def run(self):
            from data.continuous import Signal
            from data.continuous.binary import X_RGT
            Signal.set_binary()
            counters = defaultdict(build_counters)
            i, q, jobs = self._q_jobs
            for line, dst in jobs:
                tree = Tree.fromstring(line)
                dx = Signal.from_sstb(tree)
                lp, lo = dx.binary(xtype = X_RGT)
                counter = counters[dst]
                for p, o in zip(lp, lo):
                    counter.polar.update(p)
                    counter.right.update(o)
                counter.word.update(dx.word)
                counter.length[dx.max_height] += 1
                q.put(i)
            q.put((i, sum(sum(vc.length.values()) for vc in counters.values()), dict(counters)))

    corpus = []
    for mode, lines in get_sstb_trees(fpath, split_files).items():
        for line in lines:
            corpus.append((line, mode))
                
    counters = defaultdict(build_counters)
    def receive(t, qbar):
        if isinstance(t, int):
            qbar.update(t)
        else: # summary
            t, tc, vc = t
            for ds, ss in vc.items():
                ds = counters[ds]
                for dst, src in zip(ds, ss):
                    dst.update(src)
            return t, tc, 0
    mp_while(WorkerX, corpus, receive)

    def field_fn(all_counts, field, fw_rpt):
        cnt = getattr(all_counts, field)
        if field not in ('right', 'length') and (more := cnt.keys() - getattr(counters[M_TRAIN], field)):
            if field == 'word':
                count = sum(cnt[w] for w in more)
                fw_rpt.write(f'Word vocabulary has {len(more):,} types ({count:,}/{sum(cnt.values()):,} counts) not in train-set.\n')
            else:
                fw_rpt.write(f'{field.title()} has ')
                fw_rpt.write(' '.join(more) + ' not in train-set.\n')
    return post_build(save_to_dir, build_counters, VocabCounters, counters, field_fn)

def check_data(save_dir, valid_sizes):
    try:
        _, train_tok_size, all_tok_size, polar_size, right_size = valid_sizes
        if train_tok_size > all_tok_size:
            raise ValueError(f'Train vocab({all_tok_size}) should be less than corpus vocab({all_tok_size})')
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    valid_sizes = all_tok_size, polar_size, right_size
    vocab_files = ('vocab.' + x for x in 'word polar right'.split())
    return all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))


def neg_pos(head, data, _numerators, _denominators, offset):
    gr = head.label()
    pr = data.label().split('-')
    neg_set = '01'
    pos_set = '34'
    neutral = pr[0] == '2'
    fine = gr == pr[0]
    if gr in neg_set:
        ternary = pr[0] in neg_set
        binary = pr[1] in neg_set if neutral else ternary
    elif gr in pos_set:
        ternary = pr[0] in pos_set
        binary = pr[1] in pos_set if neutral else ternary
    else:
        ternary = True
        binary = None
        
    _denominators[2 + offset] += 1
    _denominators[1 + offset] += 1
    if fine:
        _numerators[2 + offset] += 1
    if ternary:
        _numerators[1 + offset] += 1
    if binary is not None:
        _denominators[0 + offset] += 1
        if binary:
            _numerators[0 + offset] += 1

from nltk.tree import Tree
from collections import namedtuple
SentimentScore = namedtuple('SentimentScore', 'D, T, Q, d, t, q')
    # 0: root_fine, 1: root_PnN 2: root_PN, 3: fine, 4: PnN 5: PN
def calc_stan_accuracy(hfname, dfname, on_error):
    
    numerators   = [0,0,0,0,0,0]
    denominators = [0,0,0,0,0,0]
    sents = []
    with open(hfname) as fh,\
         open(dfname) as fd:
        for sid, (head, data) in enumerate(zip(fh, fd)):

            warnings = []
            _numerators   = [0,0,0,0,0,0]
            _denominators = [0,0,0,0,0,0]
            head = Tree.fromstring(head)
            data = Tree.fromstring(data)
            seq_len = len(head.leaves())
            if seq_len != len(data.leaves()):
                warnings.append(f'lengths do not match vs. {len(data.leaves())}')
            for ith in range(seq_len):
                if head.leaf_treeposition(ith) != data.leaf_treeposition(ith):
                    warnings.append(f'shapes do not match at {ith}-th leaf')
                    break
            if warnings:
                on_error(f'.{sid} len={seq_len}', warnings[-1])

            neg_pos(head, data, _numerators, _denominators, 0)
            for gt, pt in zip(head.subtrees(), data.subtrees()):
                neg_pos(gt, pt, _numerators, _denominators, 3)

            scores = []
            for i, (n, d) in enumerate(zip(_numerators, _denominators)):
                numerators  [i] += n
                denominators[i] += d
                scores.append(n/d*100 if d else float('nan'))
            sents.append(scores)
    return sents, SentimentScore(*(n/d*100 if d else float('nan') for n,d in zip(numerators, denominators)))