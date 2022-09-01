C_SSTB = 'sstb'

build_params = {C_SSTB: {}}
ft_bin = {C_SSTB: 'en'}

from utils.types import false_type, true_type
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, trapezoid_height
data_type = dict(vocab_size       = vocab_size,
                 batch_size       = train_batch_size,
                 max_len          = train_max_len,
                 bucket_len       = train_bucket_len,
                 sort_by_length   = false_type,
                 nil_as_pads      = true_type,
                 nil_is_neutral   = true_type,
                 trapezoid_height = trapezoid_height)

from data.io import make_call_fasttext, check_fasttext
from utils.types import M_TRAIN, M_TEST, M_DEVEL, UNK, NIL, num_threads
call_fasttext = make_call_fasttext(ft_bin)

split_files = {'train': M_TRAIN, 'dev': M_DEVEL, 'test': M_TEST}

from sys import stderr
from nltk.tree import Tree
from collections import Counter, namedtuple, defaultdict
from os.path import join
from data.io import check_vocab, post_build

VocabCounters = namedtuple('VocabCounters', 'length, word, polar, xtype')
build_counters = lambda: VocabCounters(Counter(), Counter(), Counter(), Counter())

def build(save_to_dir, stree_path, corp_name, verbose = True, **kwargs):
    assert corp_name == C_SSTB
            
    from time import sleep
    from data.io import distribute_jobs
    from multiprocessing import Process, Queue
    from utils.types import num_threads
    from utils.str_ops import StringProgressBar
    from utils.shell_io import byte_style

    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._q_jobs = args

        def run(self):
            from data.continuous.binary import BinarySignal
            BinarySignal.share_cells()
            counters = defaultdict(build_counters)
            i, q, jobs = self._q_jobs
            for line, dst in jobs:
                tree = Tree.fromstring(line)
                dx = BinarySignal.from_sstb(tree)
                for _, p, o in dx.signals():
                    counters[dst].polar.update(p)
                    counters[dst].xtype.update(o)
                counters[dst].word.update(dx.word())
                counters[dst].length[dx.max_height] += 1
                q.put(i)
            q.put((i, dict(counters)))

    corpus = []
    for src, dst in split_files.items():
        with open(join(stree_path, f'{src}.txt')) as fr:
            for line in fr:
                corpus.append((line, dst))
                
    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()
    thread_join_cnt = 0
    counters = defaultdict(build_counters)
    desc = f'Collecting vocabulary from {num_threads} threads ['
    with StringProgressBar.segs(num_threads, prefix = desc, suffix = ']') as qbar:
        for i in range(num_threads):
            jobs = workers[i]
            w = WorkerX(i, q, jobs)
            w.start()
            workers[i] = w
            qbar.update(i, total = len(jobs))
            
        while True:
            if q.empty():
                sleep(0.0001)
            else:
                if isinstance(t := q.get(), int):
                    qbar.update(t)
                else: # summary
                    thread_join_cnt += 1
                    t, vc = t
                    for ds, ss in vc.items():
                        ds = counters[ds]
                        for dst, src in zip(ds, ss):
                            dst.update(src)
                    suffix = f'] {thread_join_cnt} ended.'
                    qbar.desc = desc, suffix
                    workers[t].join()
                    if thread_join_cnt == num_threads:
                        break
        suffix = '] ' + byte_style(f'✔ {len(corpus)} trees.', '2')
        qbar.desc = desc, suffix

    def field_fn(all_counts, field, fw_rpt):
        cnt = getattr(all_counts, field)
        if field not in ('xtype', 'length') and (more := cnt.keys() - getattr(counters[M_TRAIN], field)):
            if field == 'word':
                count = sum(cnt[w] for w in more)
                fw_rpt.write(f'Word vocabulary has {len(more):,} types ({count:,}/{sum(cnt.values()):,} counts) not in train-set.\n')
            else:
                fw_rpt.write(f'{field.title()} has ')
                fw_rpt.write(' '.join(more) + ' not in train-set.\n')
    return post_build(save_to_dir, build_counters, VocabCounters, counters, field_fn)

def check_data(save_dir, valid_sizes):
    try: # 53,18280,21701,6,6
        _, train_tok_size, all_tok_size, sem_size, xty_size = valid_sizes
        if train_tok_size > all_tok_size:
            raise ValueError(f'Train vocab({all_tok_size}) should be less than corpus vocab({all_tok_size})')
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    tok_file = join(save_dir, 'vocab.word')
    syn_file = join(save_dir, 'vocab.polar')
    xty_file = join(save_dir, 'vocab.xtype')
    res = check_vocab(tok_file, all_tok_size)
    res = res and check_vocab(syn_file, sem_size)
    res = res and check_vocab(xty_file, xty_size)
    return res


def neg_pos(head, data, _numerators, _denominators, offset):
    gr = head.label()
    pr = data.label()
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
        
    _denominators[0 + offset] += 1
    _denominators[1 + offset] += 1
    if fine:
        _numerators[0 + offset] += 1
    if ternary:
        _numerators[1 + offset] += 1
    if binary is not None:
        _denominators[2 + offset] += 1
        if binary:
            _numerators[2 + offset] += 1

from nltk.tree import Tree
import numpy as np
def calc_stan_accuracy(hfname, dfname, error_prefix, on_error):
    
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
                on_error(error_prefix + f'.{sid} len={seq_len}', warnings[-1])

            neg_pos(head, data, _numerators, _denominators, 0)
            for gt, pt in zip(head.subtrees(), data.subtrees()):
                neg_pos(gt, pt, _numerators, _denominators, 3)

            scores = []
            for i, (n, d) in enumerate(zip(_numerators, _denominators)):
                numerators  [i] += n
                denominators[i] += d
                scores.append(n/d*100 if d else float('nan'))
            sents.append(scores)

    scores = []
    for n,d in zip(numerators, denominators):
        scores.append(n/d*100 if d else float('nan'))
    # 0: root_fine, 1: root_PnN 2: root_PN, 3: fine, 4: PnN 5: PN
    return sents, scores, (np.asarray(numerators), np.asarray(denominators))