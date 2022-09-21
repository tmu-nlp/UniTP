C_DPTB = 'dptb'
C_TIGER = 'tiger'
E_DISCONTINUOUS = C_TIGER, C_DPTB

from data.io import make_call_fasttext, check_vocab, split_dict, post_build, get_corpus
build_params = {C_DPTB:  split_dict(   '2-21',          '22',          '23'),
                C_TIGER: split_dict('1-40474', '40475-45474', '45475-50474')}
ft_bin = {C_DPTB: 'en', C_TIGER: 'de'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import make_parse_data_config, make_parse_factor, make_beta, make_close_frac
from utils.types import true_type, ply_shuffle
from utils.types import tune_epoch_type, inter_height_2d
from utils.types import F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP, K_CORP, F_SENTENCE

def dccp_factor(*level_left_right):
    return make_parse_factor(binarization = make_beta(*level_left_right),
                             ply_shuffle  = ply_shuffle)

dccp_data_config = make_parse_data_config()
dccp_data_config[K_CORP] = {
    C_DPTB:  dccp_factor(F_SENTENCE, 1, 1),
    C_TIGER: dccp_factor(F_SENTENCE, 1, 1)
}

xccp_data_config = make_parse_data_config(
    min_gap      = tune_epoch_type,
    max_interply = inter_height_2d,
    continuous_fence_only = true_type)

factors = F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP
factors = {f: make_close_frac(float(f == F_RANDOM)) for f in factors}
def xccp_factor():
    return make_parse_factor(msub = 0.25, **factors)
xccp_data_config[K_CORP] = {
     C_DPTB: xccp_factor(),
     C_TIGER: xccp_factor()
}


from utils.shell_io import byte_style
from utils.str_ops import linebar
def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set):
    from data.cross import Signal
    Signal.set_binary()
    Signal.set_multib()
    if corp_name == C_TIGER:
        from xml.etree import ElementTree
        folder_pattern = lambda x: f's{x}'
        root = ElementTree.parse(corp_path).getroot()[1]
        _corpus = {sent.get('id'): sent for sent in linebar(root)}
        if train_set == '__rest__':
            train_set = set(_corpus)
        get_fileids = lambda sids: (_corpus[s] for s in sids if s in _corpus) # some sids are not in _corpus
        corpus = get_corpus(train_set, devel_set, test_set, folder_pattern, get_fileids)
        from_corpus = Signal.from_tiger_graph
    else:#if corp_name == C_DPTB:
        from data.penn_types import select_and_split_corpus as sasc
        from_corpus = Signal.from_disco_penn
        (reader, _corpus, _) = sasc(corp_name[1:], corp_path, train_set, devel_set, test_set)
        corpus = defaultdict(list)
        for m, fs in linebar(_corpus.items()):
            for fn in fs:
                for tree in reader.parsed_sents(fn):
                    corpus[m].append(tree)
    return corpus, from_corpus

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
    from data.mp import Rush, Process
    from data.cross.evalb_lcfrs import disco_exact_match
    from data.cross.binary import disco_tree as tree_from_binary
    from data.cross.multib import disco_tree as tree_from_multib

    fileid_split, from_corpus = select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set)
    corpus = []
    for m, trees in fileid_split.items():
        for tree in trees:
            corpus.append((m, tree))

    class WorkerX(Process):
        estimate_total = False

        def __init__(self, *args):
            Process.__init__(self)
            self._args = args

        def run(self):

            errors = []
            i, q, sents = self._args
            counters = defaultdict(build_counters)

            for ds, sent in sents:
                counter = counters[ds]
                try:
                    dx = from_corpus(sent)
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

            q.put((i, sum(sum(vc.length.values()) for vc in counters.values()), errors, dict(counters)))

    rush = Rush(WorkerX, corpus)
    errors = []
    counters = defaultdict(build_counters)
    def receive(t, qbar):
        if isinstance(t, int):
            qbar.update(t)
        else: # summary
            t, tc, er, vc = t
            for ds, ss in vc.items():
                ds = counters[ds]
                for dst, src in zip(ds, ss):
                    dst.update(src)
            errors.extend(er)
            return t, tc
    rush.mp_while(receive, 'Collecting vocabulary')

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