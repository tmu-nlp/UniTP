C_TGR = 'tiger'
C_DPTB = 'dptb'
C_ABSTRACT = 'disco'
E_DISCO = C_TGR, C_DPTB

from data.io import make_call_fasttext, check_fasttext, check_vocab, split_dict
build_params = {C_DPTB: split_dict(   '2-21',          '22',          '23'),
                C_TGR:  split_dict('1-40474', '40475-45474', '45475-50474')}
ft_bin = {C_DPTB: 'en', C_TGR: 'de'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import none_type, false_type, true_type, binarization_5_head, NIL, swapper
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, tune_epoch_type
disco_config = dict(vocab_size     = vocab_size,
                    binarization   = binarization_5_head,
                    batch_size     = train_batch_size,
                    max_len        = train_max_len,
                    bucket_len     = train_bucket_len,
                    min_gap        = tune_epoch_type,
                    shuffle_swap   = swapper,
                    unify_sub      = true_type,
                    sort_by_length = false_type)

# tree = '/Users/zchen/KK/corpora/tiger_release_aug07.corrected.16012013.xml'

from utils.file_io import join, remove, isfile, parpath, create_join
from sys import stderr
def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set,
          **kwargs):
    from data.cross import read_tiger_graph, read_disco_penn, zip_to_logit, zip_swaps
    from tqdm import tqdm
    from xml.etree import ElementTree
    from nltk.corpus import BracketParseCorpusReader
    from collections import Counter, defaultdict
    from itertools import count, tee
    from data.io import save_vocab, sort_count
    from utils.str_ops import strange_to, histo_count
    from utils.pickle_io import pickle_dump
    from utils.shell_io import byte_style
    from data.io import SourcePool, distribute_jobs
    from multiprocessing import Process, Queue
    from utils.types import E_ORIF5_HEAD, E_ORIF5, M_TRAIN, M_DEVEL, M_TEST, num_threads
    from contextlib import ExitStack
    from time import sleep
    from data.delta import logits_to_xtype

    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._args = args
        
        def run(self):
            errors   = []
            word_cnt = Counter()
            tag_cnt  = Counter()
            sent_gap_cnt   = defaultdict(int)
            phrase_gap_cnt = Counter()
            q, sents, read_func, enum_ori = self._args
            label_cnts = {o: Counter() for o in enum_ori}
            xtype_cnts = {o: Counter() for o in enum_ori}
            for sent_id, sent, sent_dep in sents:
                try:
                    wd, bt, cnf_layers, gd, lines = read_func(sent, sent_dep)
                except ValueError as e:
                    errors.append(sent_id + f' Value {e}')
                    continue
                except AssertionError as e:
                    errors.append(sent_id + f' Assert {e}')
                    continue
                except UnboundLocalError: # No
                    # if len(sent[0][0]) > 1 and len(sent[0][1]) > 0:
                    errors.append('NoNT')
                    continue

                cnf_bundle = {}
                for cnf_factor, (lls, lrs, ljs, lds, lsp) in cnf_layers.items():
                    cindex, labels, xtypes = zip_to_logit(lls, lrs, ljs, lds)
                    label_cnts[cnf_factor] += Counter(labels)
                    xtype_cnts[cnf_factor] += Counter(logits_to_xtype(x) for x in xtypes)
                    cindex = ' '.join(str(x) for x in cindex) + '\n'
                    labels = ' '.join(labels) + '\n'
                    xtypes = ' '.join(str(x) for x in xtypes) + '\n'
                    swapbl = zip_swaps(lsp) + '\n'
                    cnf_bundle[cnf_factor] = cindex, labels, xtypes, swapbl

                phrase_gap_cnt += Counter(gd.values())
                sent_gap_degree = max(gd.values())
                sent_gap_cnt[sent_gap_degree] += 1
                word_cnt += Counter(wd)
                tag_cnt  += Counter(bt)
                wd = ' '.join(wd) + '\n'
                bt = ' '.join(bt) + '\n'
                bundle = sent_id, len(wd), wd, bt, cnf_bundle, sent_gap_degree, '\n'.join(lines)
                q.put(bundle)

            bundle = len(sents) - len(errors), errors, word_cnt, tag_cnt, label_cnts, xtype_cnts, phrase_gap_cnt, sent_gap_cnt
            q.put(bundle)

    if corp_name == C_DPTB:
        folder_pattern = lambda x: f'{x:02}'
        reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")
        devel_set = strange_to(devel_set, folder_pattern)
        test_set  = strange_to(test_set,  folder_pattern)
        non_train_set = set(devel_set + test_set)
        if train_set == '__rest__':
            all_folders = reader.fileids()
            all_sets = set(fp[:2] for fp in all_folders)
            train_set = all_sets - non_train_set
        else:
            train_set = strange_to(train_set, folder_pattern)
            all_sets = non_train_set | set(train_set)
            all_folders = [fp for fp in reader.fileids() if fp[:2] in all_sets]
        in_train_set = lambda fn: fn in train_set
        in_devel_set = lambda fn: fn in devel_set
        in_test_set  = lambda fn: fn in test_set
        corpus = []
        for fp in all_folders:
            for sent in reader.parsed_sents(fp):
                corpus.append((fp[:2], sent, None))
        read_func = read_disco_penn
        enum_binarizations = E_ORIF5
    elif corp_name == C_TGR:
        from data.cross.tiger2dep import get_tiger_heads
        devel_set = strange_to(devel_set)
        test_set  = strange_to(test_set)
        non_train_set = devel_set + test_set
        in_devel_set = lambda x: int(x[1:]) in devel_set
        in_test_set  = lambda x: int(x[1:]) in test_set
        if train_set == '__rest__':
            in_corpus = lambda x: True
            in_train_set = lambda x: int(x[1:]) not in non_train_set
        else:
            train_set = strange_to(train_set)
            all_sets = train_set + non_train_set
            in_corpus = lambda x: int(x[1:]) in all_sets
            in_train_set = lambda x: int(x[1:]) in train_set
        root = ElementTree.parse(corp_path).getroot()
        dep_root = get_tiger_heads(corp_path)
        corpus = []
        for sent in root[1]:
            sid = sent.get('id')
            if in_corpus(sid):
                corpus.append((sid, sent, dep_root.pop(sid)))
        read_func = read_tiger_graph
        enum_binarizations = E_ORIF5_HEAD

    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()
    for i in range(num_threads):
        w = WorkerX(q, workers[i], read_func, enum_binarizations)
        w.start()
        workers[i] = w

    errors = []
    thread_join_cnt = 0
    word_cnt = Counter()
    tag_cnt  = Counter()
    sent_gap_cnt   = Counter()
    phrase_gap_cnt = Counter()
    label_cnts = {o: Counter() for o in enum_binarizations}
    xtype_cnts = {o: Counter() for o in enum_binarizations}
    length_stat = dict(tlc = defaultdict(int), vlc = defaultdict(int), _lc = defaultdict(int))
    gap_bin_size = 1 if corp_name == C_DPTB else 2
    lines_by_gap_degree = defaultdict(list)
    with ExitStack() as stack, tqdm(desc = f'  Receiving samples from {num_threads} threads') as qbar:
        ftw  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.word'), 'w'))
        ftp  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.tag'),  'w'))
        ftg  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.gap'),  'w'))
        fvw  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.word'), 'w'))
        fvp  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.tag'),  'w'))
        fvg  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.gap'),  'w'))
        f_w  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.word'), 'w'))
        f_p  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.tag'),  'w'))
        f_g  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.gap'),  'w'))
        ftxs = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.xtype.{o}'), 'w')) for o in enum_binarizations}
        ftls = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.label.{o}'), 'w')) for o in enum_binarizations}
        ftis = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.index.{o}'), 'w')) for o in enum_binarizations}
        ftss = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.swap.{o}' ), 'w')) for o in enum_binarizations}
        fvxs = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.xtype.{o}'), 'w')) for o in enum_binarizations}
        fvls = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.label.{o}'), 'w')) for o in enum_binarizations}
        fvis = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.index.{o}'), 'w')) for o in enum_binarizations}
        f_xs = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.xtype.{o}' ), 'w')) for o in enum_binarizations}
        f_ls = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.label.{o}' ), 'w')) for o in enum_binarizations}
        f_is = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.index.{o}' ), 'w')) for o in enum_binarizations}
        
        for _ in count(): # Yes! this edition works fine!
            if q.empty():
                sleep(0.01)
            else:
                t = q.get()
                if len(t) == 7:
                    sent_id, sent_len, wd, bt, cnf_bundle, sent_gap_degree, lines = t
                    if in_devel_set(sent_id):
                        length_stat['vlc'][sent_len] += 1
                        fw, ft, fxs, fls, fis, fss, fg = fvw, fvp, fvxs, fvls, fvis, None, fvg
                    elif in_test_set(sent_id):
                        length_stat['_lc'][sent_len] += 1
                        fw, ft, fxs, fls, fis, fss, fg = f_w, f_p, f_xs, f_ls, f_is, None, f_g
                    else:
                        assert in_train_set(sent_id)
                        length_stat['tlc'][sent_len] += 1
                        fw, ft, fxs, fls, fis, fss, fg = ftw, ftp, ftxs, ftls, ftis, ftss, ftg
                    fw.write(wd)
                    ft.write(bt)
                    fg.write(f'{sent_gap_degree}\n')
                    for cnf_factor, (cindex, labels, xtypes, swaps) in cnf_bundle.items():
                        fis[cnf_factor].write(cindex)
                        fls[cnf_factor].write(labels)
                        fxs[cnf_factor].write(xtypes)
                        if fss is not None:
                            fss[cnf_factor].write(swaps)
                    lines_by_gap_degree[sent_gap_degree // gap_bin_size].append((sent_len, sent_gap_degree, lines))
                    qbar.update(1)
                elif len(t) == 8:
                    thread_join_cnt += 1
                    ic, es, wc, tc, lcs, xcs, pgc, sgc = t
                    if qbar.total:
                        qbar.total += ic
                    else:
                        qbar.total = ic
                    errors.extend(es)
                    word_cnt += wc
                    tag_cnt  += tc
                    phrase_gap_cnt += pgc
                    sent_gap_cnt   += sgc
                    for o in enum_binarizations:
                        label_cnts[o] += lcs[o]
                        xtype_cnts[o] += xcs[o]
                    qbar.desc = f'  {thread_join_cnt} of {num_threads} threads ended with {qbar.total} samples, receiving'
                    if thread_join_cnt == num_threads:
                        break
                else:
                    raise ValueError('Unknown data: %r' % t)
        for w in workers:
            w.join()
        errors = Counter(errors)
        thread_str = byte_style(f'{num_threads}', '3')
        sample_str = byte_style(f'{qbar.total}', '2')
        error_str0 = byte_style(f'{sum(errors.values())}' if errors else 'No', '1' if errors else '2')
        error_str1 = (' from \'' + ', '.join(byte_style(x, '3') for x in errors) + '\'.') if errors else '.'
        qbar.desc = (f'All {thread_str} threads ended with {sample_str} samples, {error_str0} errors{error_str1}')
        instance_cnt = qbar.total

    non_gap = sent_gap_cnt.pop(0)
    non_gap_percentage = byte_style(f'{100 - non_gap / instance_cnt * 100:.2f} % ({instance_cnt - non_gap})', '2')
    print(f'Gap degree distribution among {non_gap_percentage} trees which contain gaps:', file = stderr)
    print(histo_count(sent_gap_cnt, bin_size = gap_bin_size), file = stderr)
    sent_gap_cnt[0] = non_gap

    fpath = create_join(save_to_dir, 'gap_trees')
    for gap_bin, len_gap_lines in lines_by_gap_degree.items():
        len_gap_lines.sort(key = lambda x: x[:2])
        with open(join(fpath, f'{gap_bin:02d}.trees.({len(len_gap_lines)})'), 'w') as fw:
            for tid, (sent_len, gap, lines) in enumerate(len_gap_lines):
                fw.write(f'{tid}, gap = {gap}\n' + lines + '\n\n')

    length_stat['gap'] = sent_gap_cnt
    length_stat['phrase.gap'] = phrase_gap_cnt
    pickle_dump(join(save_to_dir, 'info.pkl'), length_stat)
    tok_file = join(save_to_dir, 'vocab.word')
    tag_file = join(save_to_dir, 'vocab.tag' )
    xty_file = join(save_to_dir, 'vocab.xtype')
    syn_file = join(save_to_dir, 'vocab.label')
    # ftag_file = join(save_to_dir, 'vocab.ftag')
    _, ts = save_vocab(tok_file, word_cnt, [NIL])
    _, ps = save_vocab(tag_file, tag_cnt,  [NIL])
    _, ss = save_vocab(syn_file, sum(label_cnts.values(), Counter()), [NIL])
    _, xs = save_vocab(xty_file, sum(xtype_cnts.values(), Counter()))
    for o in enum_binarizations:
        save_vocab(join(save_to_dir, f'stat.xtype.{o}'), xtype_cnts[o])
        save_vocab(join(save_to_dir, f'stat.label.{o}'), label_cnts[o])
    from data.disco import DiscoReader
    try:
        DiscoReader(save_to_dir, None, True)
    except (KeyboardInterrupt, Exception, AssertionError) as err:
        print(err, file = stderr)
    return (ts, ps, xs, ss)

def check_data(save_dir, valid_sizes):
    try:
        ts, ps, xs, ss = valid_sizes
    except Exception as e:
        from sys import stderr
        print(e, file = stderr)
        return False
    valid_sizes = ts, ps, xs, ss
    vocab_files = 'vocab.word vocab.tag vocab.xtype vocab.label'.split()
    from data.io import check_vocab
    return all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))