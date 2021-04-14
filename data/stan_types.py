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

from nltk.tree import Tree
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.pickle_io import pickle_dump, pickle_load
from utils.str_ops import histo_count
from os.path import join, isfile
from data.io import check_vocab, save_vocab, sort_count
from sys import stderr
from data.delta import DeltaX, lnr_order, xtype_to_logits
from unidecode import unidecode

def string_to_word_polar_xtype(line):
    line = line.replace(b'\\/', b'/').replace(b'\xc2\xa0', b'.').decode('utf-8')
    line = unidecode(line)
    tree = Tree.fromstring(line)
    x    = DeltaX.from_stan(tree)
    polar, direc, _ = x.to_triangles()
    words = tree.leaves()
    polar = tuple(p[0] if isinstance(p, list) else p for p in polar)
    return words, polar, direc

def build(save_to_dir, stree_path, corp_name, verbose = True, **kwargs):
    assert corp_name == C_SSTB

    jobs = []
    for src, dst in split_files.items():
        with open(join(stree_path, f'{src}.txt'), 'rb') as fr:
            for line in fr:
                jobs.append((line, dst))
            
    from data.io import distribute_jobs
    from multiprocessing import Process, Queue
    from utils.types import num_threads
    num_threads = min(num_threads, len(jobs))
    workers = distribute_jobs(jobs, num_threads)
    q = Queue()
    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._q_jobs = args

        def run(self):
            q, jobs = self._q_jobs
            len_cnts = {}
            tok_cnt, pol_cnt, xty_cnt = Counter(), Counter(), Counter()
            for line, dst in jobs:
                token, polar, xtype = string_to_word_polar_xtype(line) # terrible sign
                tok_cnt += Counter(token)
                pol_cnt += Counter(polar)
                xty_cnt += Counter(xtype)
                xtype = tuple(xtype_to_logits(x) for x in xtype)
                instance = token, polar, xtype, dst
                q.put(instance)
                if dst not in len_cnts:
                    len_cnts[dst] = Counter()
                len_cnts[dst][len(token)] += 1
            summary = tok_cnt, pol_cnt, xty_cnt, len_cnts, dst
            q.put(summary)

    for i in range(num_threads):
        w = WorkerX(q, workers[i])
        w.start()
        workers[i] = w

    train_tok_cnt = thread_join_cnt = 0
    tok_cnt, pol_cnt, xty_cnt = Counter(), Counter(), Counter()
    len_cnts = {dst: Counter() for dst in split_files.values()}
    from contextlib import ExitStack
    from itertools import count
    from time import sleep
    with ExitStack() as stack, tqdm(desc = f'  Receiving samples from {num_threads} threads', total = len(jobs)) as qbar:
        ttf = stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.word'),  'w', encoding = 'utf-8'))
        ptf = stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.polar'), 'w', encoding = 'utf-8'))
        xtf = stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.xtype'), 'w', encoding = 'utf-8'))
        tvf = stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.word'),  'w', encoding = 'utf-8'))
        pvf = stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.polar'), 'w', encoding = 'utf-8'))
        xvf = stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.xtype'), 'w', encoding = 'utf-8'))
        t_f = stack.enter_context(open(join(save_to_dir, f'{M_TEST}.word'),   'w', encoding = 'utf-8'))
        p_f = stack.enter_context(open(join(save_to_dir, f'{M_TEST}.polar'),  'w', encoding = 'utf-8'))
        x_f = stack.enter_context(open(join(save_to_dir, f'{M_TEST}.xtype'),  'w', encoding = 'utf-8'))
        for instance_cnt in count(): # Yes! this edition works fine!
            if q.empty():
                sleep(0.0001)
            else:
                t = q.get()
                if len(t) == 4:
                    token, polar, xtype, dst = t
                    if dst == M_TRAIN:
                        tf, pf, xf = ttf, ptf, xtf
                    elif dst == M_DEVEL:
                        tf, pf, xf = tvf, pvf, xvf
                    else:
                        tf, pf, xf = t_f, p_f, x_f
                    tf.write(' '.join(token) + '\n')
                    pf.write(' '.join(polar) + '\n')
                    xf.write(' '.join(xtype) + '\n')
                    qbar.update(1)
                else: # summary
                    thread_join_cnt += 1
                    tc, pc, xc, lcs, dst = t
                    if dst == M_TRAIN:
                        train_tok_cnt += len(tc)
                    tok_cnt += tc
                    pol_cnt += pc
                    xty_cnt += xc
                    for _dst, lc in lcs.items():
                        len_cnts[_dst] += lc
                    if thread_join_cnt == num_threads:
                        break

    for dst, len_cnt in len_cnts.items():
        print(f'Length distribution in [ {dst.title()} set ]', file = stderr)
        print(histo_count(len_cnt, bin_size = 10), file = stderr)
    for w in workers:
        w.join()

    pickle_dump(join(save_to_dir, 'info.pkl'), len_cnts)

    tok_file = join(save_to_dir, 'vocab.word')
    pol_file = join(save_to_dir, 'vocab.polar')
    xty_file = join(save_to_dir, 'vocab.xtype')
    _, ts = save_vocab(tok_file, tok_cnt, [NIL])
    _, ss = save_vocab(pol_file, pol_cnt, [NIL])
    _, dr = save_vocab(xty_file, xty_cnt, lnr_order(xty_cnt)[0])
    return train_tok_cnt, ts, ss, dr

def check_data(save_dir, valid_sizes):
    try:
        _, tok_size, syn_size, xty_size = valid_sizes
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    tok_file = join(save_dir, 'vocab.word')
    syn_file = join(save_dir, 'vocab.polar')
    xty_file = join(save_dir, 'vocab.xtype')
    res = check_vocab(tok_file, tok_size)
    res = res and check_vocab(syn_file, syn_size)
    res = res and check_vocab(xty_file, xty_size)

    fname = join(save_dir, 'info.pkl')
    if res and isfile(fname):
        info = pickle_load(fname)
        print('Total:', sum(sum(info[ds].values()) for ds in split_files.values()), file = stderr)
        for ds in split_files.values():
            print(f'Length distribution in [ {ds.title()} set ] ({sum(info[ds].values())})', file = stderr)
            print(histo_count(info[ds], bin_size = 10), file = stderr)
    
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