C_DPTB = 'dptb'
C_TIGER = 'tiger'
E_DISCONTINUOUS = C_TIGER, C_DPTB

from data.io import make_call_fasttext, check_vocab, split_dict, post_build, get_corpus
build_params = {C_DPTB:  split_dict(   '2-21',          '22',          '23'),
                C_TIGER: split_dict('1-40474', '40475-45474', '45475-50474')}
ft_bin = {C_DPTB: 'en', C_TIGER: 'de'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import make_parse_data_config, make_parse_factor, make_beta, make_close_frac
from utils.types import true_type, ply_shuffle, tune_epoch_type
from utils.types import tune_epoch_type, inter_height_2d
from utils.types import F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP, K_CORP, F_SENTENCE

def dccp_factor(*level_left_right):
    return make_parse_factor(
        min_gap = tune_epoch_type,
        binarization = make_beta(*level_left_right),
        ply_shuffle  = ply_shuffle)

dccp_data_config = make_parse_data_config()
dccp_data_config['nil_pad'] = true_type
dccp_data_config[K_CORP] = {
    C_DPTB:  dccp_factor(F_SENTENCE, 1, 1),
    C_TIGER: dccp_factor(F_SENTENCE, 1, 1)
}

xccp_data_config = make_parse_data_config(
    continuous_chunk_only = true_type)

factors = F_CON, F_RANDOM, F_LEFT, F_RIGHT, F_DEP
factors = {f: make_close_frac(float(f == F_RANDOM)) for f in factors}
def xccp_factor():
    return make_parse_factor(
        msub = 0.25,
        min_gap = tune_epoch_type,
        max_interply = inter_height_2d,
        medoid = factors,
        disco_2d = dict(intra_rate = make_close_frac(0.01),
                        inter_rate = make_close_frac(1.0)))
xccp_data_config[K_CORP] = {
     C_DPTB: xccp_factor(),
     C_TIGER: xccp_factor()
}


from utils.shell_io import byte_style
def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set):
    if corp_name == C_TIGER:
        from xml.etree import ElementTree
        folder_pattern = lambda x: f's{x}'
        root = ElementTree.parse(corp_path).getroot()[1]
        _corpus = {sent.get('id'): sent for sent in root}
        if train_set == '__rest__':
            train_set = set(_corpus)
        get_fileids = lambda sids: (_corpus[s] for s in sids if s in _corpus) # some sids are not in _corpus
        return get_corpus(train_set, devel_set, test_set, folder_pattern, get_fileids)

from collections import Counter, defaultdict, namedtuple
VocabCounters = namedtuple('VocabCounters', 'length, word, tag, label, sentence_gap, phrase_gap')
build_counters = lambda: VocabCounters(Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
from utils.file_io import join, parpath
from sys import stderr

def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set):

    from data.mp import mp_while, Process
    from data.cross import Signal, draw_str_lines
    from data.cross.evalb_lcfrs import read_param
    prm_args = read_param(join(parpath(__file__, 2), 'discodop.prm'))
    Signal.set_binary()
    Signal.set_multib()
    if corp_name == C_TIGER:
        corpus = [(m, tree) for m, trees in select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set).items() for tree in trees]
        class WorkerX(Process):
            estimate_total = False

            def __init__(self, *args):
                Process.__init__(self)
                self._args = args

            def run(self):
                errors = []
                i, q, sents = self._args
                counters = defaultdict(build_counters)
                gap_lines = defaultdict(list)

                for ds, sent in sents:
                    if res := sample_process(Signal.from_tiger_graph, sent, errors, counters[ds], prm_args):
                        sent_gap, (bt, td) = res
                        gap_lines[sent_gap].append(draw_str_lines(bt, td))
                    q.put(i)
                q.put((i, sum(sum(vc.length.values()) for vc in counters.values()), errors, dict(counters), gap_lines))
        extra = ()
    else:
        from data.penn_types import select_and_split_corpus as sasc
        (reader, corpus, _) = sasc(corp_name[1:], corp_path, train_set, devel_set, test_set)
        corpus = [(m, fn) for m, fns in corpus.items() for fn in fns]
        extra = reader,
        class WorkerX(Process):
            estimate_total = True

            def __init__(self, *args):
                Process.__init__(self)
                self._args = args

            def run(self):
                i, q, fileids, reader = self._args
                n_proceed = l_estimate = 0
                n_fileid = len(fileids)
                counters = defaultdict(build_counters)
                gap_lines = defaultdict(list)
                errors = []

                for eid, (ds, fn) in enumerate(fileids):
                    trees = reader.parsed_sents(fn)
                    n_proceed += len(trees)
                    n_estimate = int(n_proceed * n_fileid / (eid + 1))
                    counter = counters[ds]

                    for tree in trees:
                        if res := sample_process(Signal.from_disco_penn, tree, errors, counter, prm_args):
                            sent_gap, (bt, td) = res
                            gap_lines[sent_gap].append(draw_str_lines(bt, td))
                        q.put((i, n_estimate) if n_estimate != l_estimate else i)
                q.put((i, sum(sum(vc.length.values()) for vc in counters.values()), errors, dict(counters), gap_lines))

    errors = []
    counters = defaultdict(build_counters)
    gap_lines = defaultdict(list)
    def receive(t, qbar):
        if isinstance(t, int):
            qbar.update(t)
        elif len(t) == 2:
            i, t = t
            qbar.update(i, total = t)
            qbar.update(i)
        else: # summary
            t, tc, er, vc, gl = t
            for ds, ss in vc.items():
                ds = counters[ds]
                for dst, src in zip(ds, ss):
                    dst.update(src)
            for g, l in gl.items():
                gap_lines[g] += l
            errors.extend(er)
            return t, tc, 0 # explain outside receive
    mp_while(WorkerX, corpus, receive, *extra)

    for sent_gap in sorted(gap_lines):
        lines = gap_lines[sent_gap]
        lines.sort(key = lambda x: len(x[0]))
        with open(join(save_to_dir, f'gap.{sent_gap}.({len(lines)}).art'), 'w') as fw:
            for eid, lines in enumerate(lines):
                fw.write(f'{eid}.\n')
                fw.write('\n'.join(lines) + '\n\n')

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
        ls, tts, ats, ts, lbs, sg, pg = valid_sizes
        if tts > ats:
            raise ValueError(f'Train vocab({tts}) should be less than corpus vocab({ats})')
        if sg > pg:
            raise ValueError(f'Gap counts at both sentence ({sg}) and phrase ({pg}) levels should be equal.')
    except Exception as e:
        from sys import stderr
        print(e, file = stderr)
        return False
    valid_sizes = ls, ats, ts, lbs, sg, pg
    vocab_files = ('vocab.' + x for x in 'length word tag label sentence_gap phrase_gap'.split())
    return all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))

from data.cross.evalb_lcfrs import disco_exact_match
from data.cross.binary import disco_tree as tree_from_binary
from data.cross.multib import disco_tree as tree_from_multib
def sample_process(from_corpus, sent, errors, counter, prm_args):
    try:
        dx = from_corpus(sent)
    except ValueError as e:
        errors.append(f' Value {e}')
        return
    except AssertionError as e:
        errors.append(f' Assert {e}')
        return
    except UnboundLocalError: # No
        # if len(sent[0][0]) > 1 and len(sent[0][1]) > 0:
        errors.append('NoNT')
        return
    wd = dx.word
    tg = dx.tag
    bl, bx, bj, _ = dx.binary()
    ml, ms, _ = dx.multib()
    bbt, btp, _ = tree_from_binary(wd, tg, bl, bx, bj)
    mbt, mtp, _ = tree_from_multib(wd, tg, ml, ms)
    if disco_exact_match(bbt, btp, mbt, mtp):
        counter.length[len(wd)] += 1
        counter.word.update(wd)
        counter.tag .update(tg)
        gaps = dx.gaps(prm_args).values()
        sent_gap = max(gaps)
        for l in bl + ml:
            counter.label.update(l)
        counter.sentence_gap[sent_gap] += 1
        counter.phrase_gap.update(gaps)
        return sent_gap, dx.tree
    else:
        errors.append('BnM')