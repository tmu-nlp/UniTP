C_TGR = 'tiger'
C_DPTB = 'dptb'
C_ABSTRACT = 'disco'
E_DISCO = C_TGR, C_DPTB

from data.delta import add_efficient_subs
from data.io import make_call_fasttext, check_fasttext, check_vocab, split_dict
build_params = {C_DPTB: split_dict(   '2-21',          '22',          '23'),
                C_TGR:  split_dict('1-40474', '40475-45474', '45475-50474')}
ft_bin = {C_DPTB: 'en', C_TGR: 'de'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import false_type, true_type, binarization_5_head, NIL, swapper, frac_close_0, frac_close_1
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, tune_epoch_type, inter_height_2d
dccp_data_config = dict(vocab_size     = vocab_size,
                        binarization   = binarization_5_head,
                        batch_size     = train_batch_size,
                        max_len        = train_max_len,
                        bucket_len     = train_bucket_len,
                        min_gap        = tune_epoch_type,
                        shuffle_swap   = swapper,
                        add_efficient_subs = true_type,
                        unify_sub      = true_type,
                        sort_by_length = false_type)

from data.cross.multib import F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP
medium_factor = dict(balanced = frac_close_0,
                     more_sub = frac_close_0,
                     others = {F_RANDOM: frac_close_1,
                               F_LEFT: frac_close_0,
                               F_RIGHT: frac_close_0,
                               F_DEP: frac_close_0,
                               F_CON: frac_close_0})
xccp_data_config = dict(vocab_size     = vocab_size,
                        batch_size     = train_batch_size,
                        medium_factor  = medium_factor,
                        max_len        = train_max_len,
                        bucket_len     = train_bucket_len,
                        min_gap        = tune_epoch_type,
                        sort_by_length = false_type,
                        unify_sub      = true_type,
                        max_inter_height = inter_height_2d,
                        continuous_fence_only = true_type)
# tree = '/Users/zchen/KK/corpora/tiger_release_aug07.corrected.16012013.xml'

from utils.shell_io import byte_style
from time import sleep, time
def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set,
                            binary = True, read_dep = None):
    from utils.str_ops import strange_to
    from data.cross.binary import read_tiger_graph, read_disco_penn
    from data.cross.multib import TreeKeeper
    if corp_name == C_DPTB:
        from data.cross.ptb2dep import StanfordDependencies
        from nltk.corpus import BracketParseCorpusReader
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
        dtrees = []
        for fp in all_folders:
            for tree in reader.parsed_sents(fp):
                corpus.append([fp[:2], tree] if read_dep else (fp[:2], tree, None))
                dtrees.append(tree)
        if read_dep:
            print(byte_style('Acquiring PTB head info from Stanford CoreNLP ...', '3'), file = stderr)
            start = time()
            dep_root = StanfordDependencies(read_dep, print_func = lambda x: print(byte_style(x, dim = True), file = stderr)).convert_corpus(dtrees)
            start = time() - start
            print('  finished in ' + byte_style(f'{start:.0f}', 3) + ' sec. (' + byte_style(f'{len(corpus) / start:.2f}', '3')+ ' sents/sec.)', file = stderr)
            for duet, dep in zip(corpus, dep_root):
                duet.append(dep)
        read_func = read_disco_penn if binary else TreeKeeper.from_disco_penn
    elif corp_name == C_TGR:
        from xml.etree import ElementTree
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
        if read_dep:
            print(byte_style('Building Tiger head info with Tiger2Dep ...', '3'), file = stderr)
            start = time()
            dep_root = get_tiger_heads(corp_path)
            start = time() - start
            print('  finished in ' + byte_style(f'{start:.0f}', 3) + ' sec. (' + byte_style(f'{len(root[1]) / start:.2f}', '3')+ ' sents/sec.)', file = stderr)
        corpus = []
        for sent in root[1]:
            sid = sent.get('id')
            if in_corpus(sid):
                corpus.append((sid, sent, dep_root.pop(sid) if read_dep else None))
        read_func = read_tiger_graph if binary else TreeKeeper.from_tiger_graph
    return corpus, in_train_set, in_devel_set, in_test_set, read_func

from utils.file_io import join, parpath, create_join
from sys import stderr
def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set,
          **kwargs):
    from data.cross.binary import zip_to_logit, zip_swaps
    from tqdm import tqdm
    from collections import Counter, defaultdict
    from itertools import count
    from data.io import save_vocab, sort_count
    from utils.str_ops import histo_count
    from utils.pickle_io import pickle_dump
    from data.io import distribute_jobs
    from multiprocessing import Process, Queue
    from utils.types import E_ORIF5_HEAD, M_TRAIN, M_DEVEL, M_TEST, num_threads
    from contextlib import ExitStack
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
            q, sents, read_func = self._args
            label_cnts = {o: Counter() for o in E_ORIF5_HEAD}
            xtype_cnts = {o: Counter() for o in E_ORIF5_HEAD}
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

    (corpus, in_train_set, in_devel_set, in_test_set,
     read_func) = select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set, read_dep = parpath(save_to_dir))

    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()
    for i in range(num_threads):
        w = WorkerX(q, workers[i], read_func)
        w.start()
        workers[i] = w

    errors = []
    thread_join_cnt = 0
    word_cnt = Counter()
    tag_cnt  = Counter()
    sent_gap_cnt   = Counter()
    phrase_gap_cnt = Counter()
    label_cnts = {o: Counter() for o in E_ORIF5_HEAD}
    xtype_cnts = {o: Counter() for o in E_ORIF5_HEAD}
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
        ftxs = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.xtype.{o}'), 'w')) for o in E_ORIF5_HEAD}
        ftls = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.label.{o}'), 'w')) for o in E_ORIF5_HEAD}
        ftis = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.index.{o}'), 'w')) for o in E_ORIF5_HEAD}
        ftss = { o:stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.swap.{o}' ), 'w')) for o in E_ORIF5_HEAD}
        fvxs = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.xtype.{o}'), 'w')) for o in E_ORIF5_HEAD}
        fvls = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.label.{o}'), 'w')) for o in E_ORIF5_HEAD}
        fvis = { o:stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.index.{o}'), 'w')) for o in E_ORIF5_HEAD}
        f_xs = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.xtype.{o}' ), 'w')) for o in E_ORIF5_HEAD}
        f_ls = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.label.{o}' ), 'w')) for o in E_ORIF5_HEAD}
        f_is = { o:stack.enter_context(open(join(save_to_dir, f'{M_TEST}.index.{o}' ), 'w')) for o in E_ORIF5_HEAD}
        
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
                    assert all(x in cnf_bundle for x in E_ORIF5_HEAD), 'Missing ' + ', '.join(set(E_ORIF5_HEAD) - cnf_bundle.keys()) + f' in {sent_id}'
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
                    for o in E_ORIF5_HEAD:
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
    for o in E_ORIF5_HEAD:
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