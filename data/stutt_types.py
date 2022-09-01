C_TIGER = 'tiger'
C_DPTB = 'dptb'
C_ABSTRACT = 'discontinuous'
E_DISCONTINUOUS = C_TIGER, C_DPTB

from utils.param_ops import change_key
def select_corpus(data_config, corpus_name):
    assert C_ABSTRACT in data_config
    if corpus_name is None:
        config = change_key(data_config, C_ABSTRACT, *E_DISCONTINUOUS)
        # if 'binarization' in config:
        #     for corp_name, binarization in multilingual_binarization.items():
        #         data_config[corp_name]['binarization'] = binarization
    else:
        assert corpus_name in E_DISCONTINUOUS
        change_key(data_config, C_ABSTRACT, corpus_name)

from data.io import make_call_fasttext, check_fasttext, check_vocab, split_dict, post_build, get_corpus
build_params = {C_DPTB:  split_dict(   '2-21',          '22',          '23'),
                C_TIGER: split_dict('1-40474', '40475-45474', '45475-50474')}
ft_bin = {C_DPTB: 'en', C_TIGER: 'de'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import false_type, true_type, binarization_5_head, NIL, ply_shuffle, frac_close_0, frac_close_1, frac_25
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, tune_epoch_type, inter_height_2d
dccp_data_config = dict(vocab_size     = vocab_size,
                        binarization   = binarization_5_head,
                        batch_size     = train_batch_size,
                        max_len        = train_max_len,
                        bucket_len     = train_bucket_len,
                        min_gap        = tune_epoch_type,
                        ply_shuffle    = ply_shuffle,
                        unify_sub      = true_type,
                        sort_by_length = false_type)

from data.cross.multib import F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP
medium_factor = dict(balanced = frac_close_0,
                     more_sub = frac_25,
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


from utils.shell_io import byte_style
from utils.str_ops import StringProgressBar, linebar
from time import sleep
def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set):
    if corp_name == C_TIGER:
        from xml.etree import ElementTree
        folder_pattern = lambda x: f's{x}'
        root = ElementTree.parse(corp_path).getroot()[1]
        _corpus = {sent.get('id'): sent for sent in linebar(root)}
        if train_set == '__rest__':
            train_set = set(_corpus)
        get_fileids = lambda sids: (_corpus[s] for s in sids if s in _corpus) # some sids are not in _corpus
        corpus = get_corpus(train_set, devel_set, test_set, folder_pattern, get_fileids)
    else:#if corp_name == C_DPTB:
        from data.penn_types import select_and_split_corpus as sasc
        (reader, _corpus) = sasc(corp_name[1:], corp_path, train_set, devel_set, test_set)
        corpus = []
        for m, fn in linebar(_corpus):
            for tree in reader.parsed_sents(fn):
                corpus.append((m, tree))
    return corpus

from collections import Counter, defaultdict, namedtuple
VocabCounters = namedtuple('VocabCounters', 'length, word, tag, b_label, m_label, sentence_gap, phrase_gap')
build_counters = lambda: VocabCounters(Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
from utils.file_io import join
from sys import stderr

def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set,
          **kwargs):
    from data.io import distribute_jobs
    from multiprocessing import Process, Queue
    from utils.types import num_threads
    from data.cross.evalb_lcfrs import disco_exact_match
    from data.cross.binary import disco_tree as tree_from_binary
    from data.cross.multib import disco_tree as tree_from_multib

    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._args = args

        def run(self):
            from data.cross import Signal
            Signal.set_binary()
            Signal.set_multib()

            errors = []
            i, q, sents, corp_name = self._args
            counters = defaultdict(build_counters)
            if corp_name == C_TIGER:
                signal_cls = Signal.from_tiger_graph
            else:
                signal_cls = Signal.from_disco_penn

            for ds, sent in sents:
                counter = counters[ds]
                try:
                    dx = signal_cls(sent)
                    wd, tg = dx.word_tag
                except ValueError as e:
                    errors.append(f' Value {e}')
                    q.put(i)
                    continue
                except AssertionError as e:
                    errors.append(f' Assert {e}')
                    q.put(i)
                    continue
                except UnboundLocalError: # No
                    # if len(sent[0][0]) > 1 and len(sent[0][1]) > 0:
                    errors.append('NoNT')
                    q.put(i)
                    continue

                safe_conversion = True
                try:
                    bl, bo, bj, bd, _ = dx.binary()
                    bbt, btp, _ = tree_from_binary(wd, tg, bl, bo, bj, bd)
                    ml, ms, _ = dx.multib()
                    mbt, mtp, _ = tree_from_multib(wd, tg, ml, ms)
                    if not disco_exact_match(bbt, btp, mbt, mtp):
                        safe_conversion = False
                except:
                    safe_conversion = False
                if safe_conversion:
                    counter.length[len(wd)] += 1
                    counter.word.update(wd)
                    counter.tag .update(tg)
                    for l in bl:
                        counter.b_label.update(l)
                    for l in ml:
                        counter.m_label.update(l)
                    counter.sentence_gap[dx.gap] += 1
                    counter.phrase_gap.update(dx.gaps.values())
                else:
                    errors.append('BnM')
                q.put(i)

            q.put((i, errors, dict(counters)))

    corpus = select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set)

    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()

    errors = []
    thread_join_cnt = 0
    counters = defaultdict(build_counters)
    desc = f'Collecting vocabulary from {num_threads} threads ['
    with StringProgressBar.segs(num_threads, prefix = desc, suffix = ']') as qbar:
        for i in range(num_threads):
            jobs = workers[i]
            w = WorkerX(i, q, jobs, corp_name)
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
                    t, er, vc = t
                    for ds, ss in vc.items():
                        ds = counters[ds]
                        for dst, src in zip(ds, ss):
                            dst.update(src)
                    suffix = f'] {thread_join_cnt} ended.'
                    qbar.desc = desc, suffix
                    workers[t].join()
                    errors.extend(er)
                    if thread_join_cnt == num_threads:
                        break
        suffix = '] ' + byte_style(f'✔ {len(corpus) - len(errors)} trees.', '2')
        qbar.desc = desc, suffix

    if errors:
        desc = byte_style(f'✗ {len(errors)}', '1')
        desc += ' errors during conversion (Tiger has 2 sentences tiger_release_aug07.corrected.16012013).'
        print(desc, file = stderr)

    from utils.types import M_TRAIN
    def field_fn(all_counts, field, fw_rpt):
        cnt = getattr(all_counts, field)
        # if field == 'tag':
        #     fw_rpt.write('PoS has more: ')
        #     fw_rpt.write(' '.join(cnt.keys() - part_of_speech[corp_name]) + '\n')
        #     fw_rpt.write('PoS has less: ')
        #     fw_rpt.write(' '.join(part_of_speech[corp_name] - cnt.keys()) + '\n')
        if field not in ('xtype', 'length') and (more := cnt.keys() - getattr(counters[M_TRAIN], field)):
            if field == 'word':
                count = sum(cnt[w] for w in more)
                fw_rpt.write(f'Word vocabulary has {len(more):,} types ({count:,}/{sum(cnt.values()):,} counts) not in train-set.\n')
            else:
                fw_rpt.write(f'{field.title()} has ')
                fw_rpt.write(' '.join(more) + ' not in train-set.\n')

    return post_build(save_to_dir, build_counters, VocabCounters, counters, field_fn)

def check_data(save_dir, valid_sizes):
    try: # 96,44390,46349,46,252,172,4,109
        ls, tts, ats, ts, bl, ml, sg, pg = valid_sizes
        if tts > ats:
            raise ValueError(f'Train vocab({tts}) should be less than corpus vocab({ats})')
        if ml > bl:
            raise ValueError(f'Binary labels ({bl}) should be no less than multi-branching ({ml}).')
        if sg != pg:
            raise ValueError(f'Gap counts at both sentence ({sg}) and phrase ({pg}) levels should be equal.')
    except Exception as e:
        from sys import stderr
        print(e, file = stderr)
        return False
    valid_sizes = ls, ats, ts, bl, ml, sg, pg
    vocab_files = ('vocab.' + x for x in 'length word tag b_label m_label sentence_gap phrase_gap'.split())
    return all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))

    # in_train_set = lambda fn: fn in train_set
    # in_devel_set = lambda fn: fn in devel_set
    # in_test_set  = lambda fn: fn in test_set

    #     dtrees = []
    #     for fp in all_folders:
    #         for tree in reader.parsed_sents(fp):
    #             corpus.append([fp[:2], tree] if read_dep else (fp[:2], tree, None))
    #             dtrees.append(tree)
    #     if read_dep:
    #         print(byte_style('Acquiring PTB head info from Stanford CoreNLP ...', '3'), file = stderr)
    #         start = time()
    #         dep_root = StanfordDependencies(read_dep, print_func = lambda x: print(byte_style(x, dim = True), file = stderr)).convert_corpus(dtrees)
    #         start = time() - start
    #         print('  finished in ' + byte_style(f'{start:.0f}', 3) + ' sec. (' + byte_style(f'{len(corpus) / start:.2f}', '3')+ ' sents/sec.)', file = stderr)
    #         for duet, dep in zip(corpus, dep_root):
    #             duet.append(dep)

    #     devel_set = strange_to(devel_set)
    #     test_set  = strange_to(test_set)
    #     non_train_set = devel_set + test_set
    #     in_devel_set = lambda x: int(x[1:]) in devel_set
    #     in_test_set  = lambda x: int(x[1:]) in test_set
    #     if train_set == '__rest__':
    #         in_corpus = lambda x: True
    #         in_train_set = lambda x: int(x[1:]) not in non_train_set
    #     else:
    #         train_set = strange_to(train_set)
    #         all_sets = train_set + non_train_set
    #         in_corpus = lambda x: int(x[1:]) in all_sets
    #         in_train_set = lambda x: int(x[1:]) in train_set
    #     if read_dep:
    #         print(byte_style('Building Tiger head info with Tiger2Dep ...', '3'), file = stderr)
    #         start = time()
    #         dep_root = get_tiger_heads(corp_path)
    #         start = time() - start
    #         print('  finished in ' + byte_style(f'{start:.0f}', 3) + ' sec. (' + byte_style(f'{len(root[1]) / start:.2f}', '3')+ ' sents/sec.)', file = stderr)
    #     corpus = []
    #         if in_corpus(sid):
    #             corpus.append((sid, sent, dep_root.pop(sid) if read_dep else None))
    # return corpus, read_func, , read_dep = parpath(save_to_dir)
        # from data.cross.tiger2dep import get_tiger_heads
        # from data.cross.ptb2dep import StanfordDependencies