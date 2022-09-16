from collections import defaultdict
from sys import stderr
from os.path import isfile, join, basename

SEP = '\t'

def make_call_fasttext(corp_ft_bin):
    from utils.shell_io import call_fasttext
    def inner(fasttext, path, corp_name):
        specials = []
        char_cnt = defaultdict(int)
        with open(join(path, 'vocab.word')) as fr:
            for line in fr:
                if SEP in line:
                    word, count = line.split(SEP)
                    count = int(count)
                    for char in word:
                        char_cnt[char] += count
                else:
                    specials.append(line.rstrip())
        save_vocab(join(path, 'vocab.char'), char_cnt, specials)
        for fn in ('word', 'char'):
            wfile = join(path, 'vocab.' + fn)
            vfile = join(path, fn + '.vec')
            ft_bin = corp_ft_bin[corp_name]
            ft_bin = fasttext['ft_bin'][ft_bin]
            print(f"calling fasttext for [{wfile}:{vfile}] of {corp_name} with '{basename(ft_bin)}'", file = stderr)
            call_fasttext(fasttext['path'], wfile, vfile, ft_bin, fasttext['ft_lower'])
    return inner

def check_fasttext(path):
    return __check_ft(path, 'word') and __check_ft(path, 'char')

def __check_ft(path, fname):
    wfile = join(path, fname + '.vec')
    vfile = join(path, 'vocab.' + fname)
    if isfile(wfile) and isfile(vfile):
        # import pdb; pdb.set_trace()
        from utils.file_io import count_lines
        wlen = count_lines(wfile, True) - 1
        vlen = count_lines(vfile, True)
        if wlen == vlen:
            return True
        else:
            print(f'not match', wlen, vlen)
    else:
        print('wvfiles not exist')
    return False

def get_fasttext(fname):
    import numpy as np
    return np.loadtxt(fname, dtype = np.float32)

def save_vocab(save_to_file, main_cnt, prefix = [], appendix_cnt = None):
    # [NIL ...] + [main] [<-appendix (finally no intersection with main)]
    lines = []
    main_cnt = main_cnt.copy()
    if has_appendix := appendix_cnt is not None:
        appendix_cnt = appendix_cnt.copy()
        for obj, cnt in main_cnt.items():
            if obj in appendix_cnt:
                main_cnt[obj] = appendix_cnt.pop(obj)

    for obj in prefix:
        token = obj if isinstance(obj, str) else str(obj)
        if obj in main_cnt:
            cnt = main_cnt.pop(obj)
            lines.append(token + SEP + str(cnt))
        else:
            lines.append(token)

    main_lines = []
    for obj, cnt in main_cnt.items():
        token = obj if isinstance(obj, str) else str(obj)
        main_lines.append((1, cnt, token))

    if has_appendix:
        for obj, cnt in appendix_cnt.items():
            token = obj if isinstance(obj, str) else str(obj)
            main_lines.append((0, cnt, token))

    main_lines.sort(key = lambda x: x[2])
    main_lines.sort(key = lambda x: x[:2], reverse = True)
    main_cnt = len(lines)
    for eid, (m, cnt, token) in enumerate(main_lines):
        main_lines[eid] = token + SEP + str(cnt)
        main_cnt += m
    
    with open(save_to_file, 'w', encoding = 'utf-8') as fw:
        for line in lines:
            fw.write(line + '\n')
        fw.write('\n'.join(main_lines))

    if has_appendix:
        return main_cnt, len(lines) + len(main_lines)
    return main_cnt

def post_build(save_to_dir, build_counters, vc_type, counters, field_fn = None):
    from data import NIL
    from utils.types import M_TRAIN
    from utils.str_ops import histo_count
    sizes = []
    all_counts = build_counters()
    all_files = (join(save_to_dir, f'vocab.{x}') for x in (vc_type._fields))
    with open(join(save_to_dir, 'vocab.rpt'), 'w') as fw:
        for ds, vc in counters.items():
            fw.write(f'Length distribution in [ {ds.upper()} set ] ({sum(vc.length.values())})\n')
            fw.write(histo_count(vc.length, bin_size = 10) + '\n')
            for dst, src in zip(all_counts, vc):
                dst.update(src)

        for field, fn, cnt in zip(vc_type._fields, all_files, all_counts):
            if field == 'word':
                sizes.extend(save_vocab(fn, counters[M_TRAIN].word, [NIL], cnt))
            else:
                sizes.append(save_vocab(fn, cnt, [NIL] if field != 'length' else []))
            if callable(field_fn):
                field_fn(all_counts, field, fw)
    return sizes

def load_i2vs(vocab_dir, *suffixes):
    i2vs = {}
    for suf in suffixes:
        py_v = list(gen_vocab(join(vocab_dir, f'vocab.{suf}')))
        i2vs[suf] = py_v
    return i2vs

def gen_vocab(fname):
    special_bound = None
    with open(fname, 'r') as fr:
        for idx, tok in enumerate(fr):
            tok = tok.rstrip()
            if special_bound:
                tok = tok[:tok.find(SEP)]
            elif SEP in tok:
                special_bound = idx
                tok = tok[:tok.find(SEP)]
            yield tok

def check_vocab(fname, expected_size = None):
    special_bound = None
    try:
        with open(fname, 'r') as fr:
            for idx, tok in enumerate(fr):
                if special_bound is not None:
                    if SEP not in tok:
                        print('Ill-formed', fname, file = stderr)
                        return False
                elif SEP in tok:
                    special_bound = idx
    except Exception as e:
        print(e, file = stderr)
        return False
    if expected_size and expected_size != idx + 1:
        print('Invalid size %d vs. %d' % (expected_size, idx + 1), fname, file = stderr)
        return False
    return True

def sorting_order(seq, key = lambda x: (len(x), x)):
    order = {}
    for src in sorted(range(len(seq)), key = lambda i: key(seq[i])):
        order[src] = len(order)
    return order

def sort_by_order(order, seq):
    cache = [None for _ in seq]
    for src, dst in order.items():
        cache[dst] = seq[src]
    return cache

def split_dict(tr, vl, ts, **kwargs):
    kwargs['train_set'] = tr
    kwargs['devel_set'] = vl
    kwargs['test_set' ] = ts
    return kwargs

from random import randint
class SourcePool:
    def __init__(self, src, rand = False):
        if rand:
            bound = len(src) - 1
        else:
            bound = 0, len(src)
        self._src_b = src, bound

    def __call__(self):
        src, bound = self._src_b
        if isinstance(bound, int):
            return src[randint(0, bound)]
        idx, bound = bound
        ctt = src[idx]
        idx += 1
        if idx == bound:
            idx = 0
        self._src_b = src, (idx, bound)
        return ctt

def distribute_jobs(jobs, num_workers):
    workers = [[] for i in range(num_workers)]
    pool = SourcePool(workers)
    for fileid in jobs:
        worker = pool()
        worker.append(fileid)
    del pool
    return workers

from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.str_ops import strange_to
def get_corpus(train_set, devel_set, test_set, folder_pattern, get_fileids):
    devel_set = set(strange_to(devel_set, folder_pattern))
    test_set  = set(strange_to(test_set,  folder_pattern))
    if isinstance(train_set, set):
        non_train_set = devel_set | test_set
        assert train_set & non_train_set
        train_set -= non_train_set
    else:
        train_set = set(strange_to(train_set, folder_pattern))
    assert test_set
    assert devel_set
    assert train_set
    assert not (train_set & test_set)
    assert not (devel_set & test_set)
    assert not (devel_set & train_set)
    fileids = defaultdict(set)
    for n, s in {M_TRAIN: train_set, M_DEVEL: devel_set, M_TEST: test_set}.items():
        fileids[n].update(get_fileids(s))
    return fileids