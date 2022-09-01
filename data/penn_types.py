C_PTB = 'ptb'
C_CTB = 'ctb'
C_KTB = 'ktb'
C_NPCMJ = 'npcmj'
C_ABSTRACT = 'continuous'
E_CONTINUE = C_PTB, C_CTB, C_KTB, C_NPCMJ

from utils.types import O_LFT, O_RGT
multilingual_binarization = {
    C_PTB: {O_LFT: 0.15, O_RGT: 0.85},
    C_CTB: {O_LFT: 0.2, O_RGT: 0.8},
    C_KTB: {O_LFT: 0.7, O_RGT: 0.3},
    C_NPCMJ: {O_LFT: 0.7, O_RGT: 0.3},
}

from utils.param_ops import change_key
def select_corpus(data_config, corpus_name):
    assert C_ABSTRACT in data_config
    if corpus_name is None:
        config = change_key(data_config, C_ABSTRACT, *E_CONTINUE)
        if 'binarization' in config:
            for corp_name, binarization in multilingual_binarization.items():
                data_config[corp_name]['binarization'] = binarization
    else:
        assert corpus_name in E_CONTINUE
        change_key(data_config, C_ABSTRACT, corpus_name)

from data.io import make_call_fasttext, check_fasttext, check_vocab, split_dict
build_params = {C_PTB: split_dict('2-21',             '22',      '23'    ),
                C_CTB: split_dict('001-270,440-1151', '301-325', '271-300'),
                C_KTB: split_dict('300',              '0-14',    '15-29'),
                C_NPCMJ: split_dict('300',            '0-14',    '15-29')}
                # C_KTB: dict(train_set = 'non_numeric_naming') }
ft_bin = {C_PTB: 'en', C_CTB: 'zh', C_KTB: 'ja', C_NPCMJ: 'ja'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import false_type, true_type, binarization, frac_close_0
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, trapezoid_height
nccp_data_config = dict(vocab_size       = vocab_size,
                        binarization     = binarization,
                        batch_size       = train_batch_size,
                        max_len          = train_max_len,
                        bucket_len       = train_bucket_len,
                        with_ftags       = false_type,
                        unify_sub        = true_type,
                        sort_by_length   = false_type,
                        nil_as_pads      = true_type,
                        trapezoid_height = trapezoid_height)

accp_data_config = dict(vocab_size       = vocab_size,
                        batch_size       = train_batch_size,
                        balanced         = frac_close_0,
                        max_len          = train_max_len,
                        bucket_len       = train_bucket_len,
                        unify_sub        = true_type,
                        sort_by_length   = false_type)

from utils.str_ops import StringProgressBar, cat_lines
from sys import stderr
from os.path import join, dirname
from os import listdir
from contextlib import ExitStack
from collections import Counter, defaultdict, namedtuple

from nltk.tree import Tree
from data.io import SourcePool, distribute_jobs, post_build, get_corpus
from random import seed

def gen_trees(fpath, keep_str):
    def wrap_tree():
        if keep_str:
            return cumu_string
        tree = Tree.fromstring(cumu_string)
        if tree.label() == '':
            tree = tree[0]
        return tree

    with open(fpath) as fr:
        cumu_string = None
        for line in fr:
            if not keep_str:
                line = line.rstrip()
            if not line: continue
            if line[0] == '(': # end
                if cumu_string is not None:
                    if ')' not in line: continue
                    yield wrap_tree()
                cumu_string = line # cumu_starte = eid
            elif line[0] == '<' or len(line) <= 1: # start or end
                if cumu_string is None:
                    continue # not start yet
                yield wrap_tree()
                cumu_string = None # cumu_starte = eid
            elif cumu_string is not None:
                cumu_string += line
    if cumu_string is not None:
        if keep_str:
            cumu_string += '\n'
        yield wrap_tree()

class CorpusReader:
    def __init__(self, path):
        self._path = path

    def break_corpus(self, n_frag = 300, rand_seed = 31415926, n_samples = 3):
        from tempfile import TemporaryDirectory
        from utils.shell_io import byte_style
        src_files = self.fileids()
        src_files.sort() # ensure file order
        fpath = TemporaryDirectory() # persists til pid ends
        prefix = 'Shuffle corpus ('
        suffix = ') ' + byte_style(f'{len(src_files)}', '2')
        suffix += ' into '
        suffix += byte_style(f'{n_frag} fragments', '2')
        suffix += ' with seed ('
        suffix += byte_style(f'{rand_seed}', '2') + ').'
        first = []
        count = 0
        with ExitStack() as stack, StringProgressBar(self._path, prefix = prefix, suffix = suffix) as bar:
            dst_files = [
                stack.enter_context(open(join(fpath.name, f'temp_{i:04}.az'), 'w'))
                for i in range(n_frag)
            ]
            seed(rand_seed)
            pool = SourcePool(dst_files, True)
            bar.update(total = len(src_files))
            for src_file in src_files:
                bar.update()
                for string in self.parsed_sents(src_file, True):
                    if len(first) < n_samples and len(string) < 300:
                        first.append(Tree.fromstring(string))
                    pool().write(string)
                    count += 1
            seed(None)
        from data.cross.dptb import direct_read
        from data.cross import draw_str_lines
        if n_samples:
            print(f'\nFor example, the first {n_samples} small sample(s):', file = stderr)
            for eid, tree in enumerate(first):
                lines = [f'{eid + 1}.']
                lines.extend(draw_str_lines(*direct_read(tree)))
                if eid:
                    first = cat_lines(first, lines)
                else:
                    first = lines
            print('\n'.join(first), file = stderr)
            print(f' ... {count} samples.')
        self._path = fpath

    def fileids(self):
        if isinstance(self._path, str):
            return listdir(self._path)
        return listdir(self._path.name)

    def parsed_sents(self, fileids = None, keep_str = False):
        if fileids is None:
            fileids = self.fileids()
        fpath = self._path if isinstance(self._path, str) else self._path.name
        if isinstance(fileids, str):
            fileids = [fileids]
        trees = []
        for fn in fileids:
            for tree in gen_trees(join(fpath, fn), keep_str):
                if not keep_str and not tree.label():
                    tree = tree[0]
                trees.append(tree)
        return trees

from utils.types import M_TRAIN
def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set):
    if corp_name == C_PTB:
        folder_pattern = lambda x: f'{x:02}' # 23/wsj_0000.mrg
        from nltk.corpus import BracketParseCorpusReader
        reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")
        get_fileid = dirname
    else:
        folder_pattern = lambda x: f'{x:04}' # /xxxx_0000.xx
        reader = CorpusReader(corp_path)
        if corp_name in (C_KTB, C_NPCMJ):
            reader.break_corpus(int(train_set)); train_set = '__rest__'
        get_fileid = lambda fn: fn[5:-3]

    get_fileids = lambda ds: (fn for fn in reader.fileids() if get_fileid(fn) in ds)
    if train_set == '__rest__':
        train_set = set(get_fileid(fn) for fn in reader.fileids())
    return reader, get_corpus(train_set, devel_set, test_set, folder_pattern, get_fileids)

VocabCounters = namedtuple('VocabCounters', 'length, word, tag, label, xtype')
build_counters = lambda: VocabCounters(Counter(), Counter(), Counter(), Counter(), Counter())

def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set,
          **kwargs):

    (reader, corpus) = select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set)

    from time import sleep
    from utils import do_nothing
    from utils.types import F_RAND_CON, num_threads
    from utils.str_ops import StringProgressBar
    from utils.shell_io import byte_style
    from multiprocessing import Process, Queue # seamless with threading.Thread
    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._args = args

        def run(self):
            from data.continuous import Signal
            from data.continuous.multib import get_tree_from_signals as multib_tree
            from data.continuous.binary import get_tree_from_signals as binary_tree, X_RGT
            Signal.set_binary()
            Signal.set_multib()
            i, q, reader, fileids = self._args
            inst_cnt = n_proceed = l_estimate = 0
            n_fileid = len(fileids)
            counters = defaultdict(build_counters)
            err_cnf, err_conversion = [], []
            dx_fn = {
                C_PTB: Signal.from_ptb,
                C_CTB: Signal.from_ctb,
                C_KTB: Signal.from_ktb,
                C_NPCMJ: Signal.from_ktb}[corp_name]

            for eid, (ds, fn) in enumerate(fileids):
                trees = reader.parsed_sents(fn)
                n_proceed += len(trees)
                n_estimate = int(n_proceed * n_fileid / (eid + 1))
                counter = counters[ds]

                for tree in trees:
                    br = []
                    q.put((i, n_estimate) if n_estimate != l_estimate else i)
                    try:
                        dx = dx_fn(tree)
                        bl, bx = dx.binary(F_RAND_CON)
                        ml, ms = dx.multib()
                        for xtype in bx:
                            br.append([x & X_RGT for x in xtype])
                    except:
                        err_cnf.append((fn, eid))
                        continue
                    safe_conversion = True
                    wd = dx.word
                    tg = dx.tag
                    try:
                        btree, warning = binary_tree(wd, tg, bl, br, word_fn = do_nothing)
                        if btree != tree or warning:
                            err_conversion.append((fn, eid, warning)); safe_conversion = False
                        if multib_tree(wd, tg, ml, ms, word_fn = do_nothing) != tree:
                            err_conversion.append((fn, eid, None)); safe_conversion = False
                    except:
                        err_conversion.append((fn, eid, None))
                        safe_conversion = False
                    if safe_conversion:
                        counter.word.update(wd)
                        counter.tag .update(tg)
                        for label, xtype in zip(bl, bx):
                            counter.label.update(label)
                            counter.xtype.update(xtype)
                        inst_cnt += 1
                        counter.length[dx.max_height] += 1

            q.put((inst_cnt, dict(counters), err_cnf, err_conversion))

    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()
    for i in range(num_threads):
        w = WorkerX(i, q, reader, workers[i])
        w.start()
        workers[i] = w

    err_cnf, err_conversion = [], []
    tree_cnt = thread_join_cnt = 0
    counters = defaultdict(build_counters)
    desc = f'Collecting vocabulary from {num_threads} threads ['
    with StringProgressBar.segs(num_threads, prefix = desc, suffix = ']') as qbar:
        while True:
            if q.empty():
                sleep(0.01)
            else:
                if isinstance(t := q.get(), tuple):
                    if len(t) == 2:
                        tid, n_estimate = t
                        qbar.update(tid, total = n_estimate)
                        qbar.update(tid)
                    else:
                        thread_join_cnt += 1
                        tc, vc, erc, erv = t
                        for ds, ss in vc.items():
                            ds = counters[ds]
                            for dst, src in zip(ds, ss):
                                dst.update(src)
                        tree_cnt += tc
                        err_cnf.extend(erc)
                        err_conversion.extend(erv)
                        suffix = f'] {thread_join_cnt} ended.'
                        qbar.desc = desc, suffix
                        if thread_join_cnt == num_threads:
                            break
                else:
                    qbar.update(t)
        for w in workers:
            w.join()
        suffix = '] ' + byte_style(f'✔ {tree_cnt} trees.', '2')
        qbar.desc = desc, suffix
    if err_cnf:
        desc = byte_style(f'✗ {len(err_cnf)}', '1')
        desc += ' errors during CNF (KTB/NPCMJ has 3 sentences, e.g., \'* *** * ***\', in ver. 202202).'
        print(desc, file = stderr)
    if err_conversion:
        desc = byte_style(f'✗ {len(err_conversion)}', '1')
        desc += ' errors during checking conversion.'
        print(desc, file = stderr)

    def field_fn(all_counts, field, fw_rpt):
        cnt = getattr(all_counts, field)
        if field == 'tag':
            fw_rpt.write('PoS has more: ')
            fw_rpt.write(' '.join(cnt.keys() - part_of_speech[corp_name]) + '\n')
            fw_rpt.write('PoS has less: ')
            fw_rpt.write(' '.join(part_of_speech[corp_name] - cnt.keys()) + '\n')
        if field not in ('xtype', 'length') and (more := cnt.keys() - getattr(counters[M_TRAIN], field)):
            if field == 'word':
                count = sum(cnt[w] for w in more)
                fw_rpt.write(f'Word vocabulary has {len(more):,} types ({count:,}/{sum(cnt.values()):,} counts) not in train-set.\n')
            else:
                fw_rpt.write(f'{field.title()} has ')
                fw_rpt.write(' '.join(more) + ' not in train-set.\n')

    return post_build(save_to_dir, build_counters, VocabCounters, counters, field_fn)


def check_data(save_dir, valid_sizes):
    try:
        ls, tts, ats, ts, ss, xs = valid_sizes
        if tts > ats:
            raise ValueError(f'Train vocab({tts}) should be less than corpus vocab({ats})')
    except Exception as e:
        print(e, file = stderr)
        return False
    valid_sizes = ls, ats, ts, ss, xs
    vocab_files = ('vocab.' + x for x in 'length word tag label xtype'.split())
    return all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))

def get_tags(string):
    tags = set()
    for line in string.split('\n')[1:]:
        _, tag, _ = line.split('\t')
        assert ' ' not in tag
        tags.add(tag)
    return tags

part_of_speech = {
    C_PTB: get_tags('''
        1.	CC	Coordinating conjunction
        2.	CD	Cardinal number
        3.	DT	Determiner
        4.	EX	Existential there
        5.	FW	Foreign word
        6.	IN	Preposition or subordinating conjunction
        7.	JJ	Adjective
        8.	JJR	Adjective, comparative
        9.	JJS	Adjective, superlative
        10.	LS	List item marker
        11.	MD	Modal
        12.	NN	Noun, singular or mass
        13.	NNS	Noun, plural
        14.	NNP	Proper noun, singular
        15.	NNPS	Proper noun, plural
        16.	PDT	Predeterminer
        17.	POS	Possessive ending
        18.	PRP	Personal pronoun
        19.	PRP$	Possessive pronoun
        20.	RB	Adverb
        21.	RBR	Adverb, comparative
        22.	RBS	Adverb, superlative
        23.	RP	Particle
        24.	SYM	Symbol
        25.	TO	to
        26.	UH	Interjection
        27.	VB	Verb, base form
        28.	VBD	Verb, past tense
        29.	VBG	Verb, gerund or present participle
        30.	VBN	Verb, past participle
        31.	VBP	Verb, non-3rd person singular present
        32.	VBZ	Verb, 3rd person singular present
        33.	WDT	Wh-determiner
        34.	WP	Wh-pronoun
        35.	WP$	Possessive wh-pronoun
        36.	WRB	Wh-adverb'''), # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    C_CTB: get_tags('''
        1.	AD	adverb
        2.	AS	aspect marker
        3.	BA	把, 将 in ba-construction
        4.	CC	coordinating conjunction
        5.	CD	cardinal number
        6.	CS	subordinating conjunction
        7.	DEC	的 in a relative-clause
        8.	DEG	associative 的
        9.	DER	得 in V-de const. and V-de-R
        10.	DEV	地 before VP
        11.	DT	determiner
        12.	ETC	for words 等, 等等
        13.	FW	foreign words
        14.	IJ	interjection
        15.	JJ	other noun-modifier
        16.	LB	被 in long bei-const
        17.	LC	localizer
        18.	M	measure word
        19.	MSP	other particle
        20.	NN	common noun
        21.	NR	proper noun
        22.	NT	temporal noun
        23.	OD	ordinal number
        24.	ON	onomatopoeia , 
        25.	P	preposition excl. 被 and 把
        26.	PN	pronoun
        27.	PU	punctuation
        28.	SB	被 in short bei-const
        29.	SP	sentence-final particle
        30.	VA	predicative adjective
        31.	VC	是
        32.	VE	有 as the main verb
        33.	VV	other verb'''), # https://catalog.ldc.upenn.edu/docs/LDC2009T24/treebank/chinese-treebank-postags.pdf
    C_KTB: get_tags('''
        1.	PUQ	quote, offically claimed as QUOT
        2.	PUL	left bracket, offically claimed as -LRB-
        3.	PUR	right bracket, offically claimed as -RRB-
        4.	PU	punctuation
        5.	ADJI	い-adjective
        6.	ADJN	な-adjective
        7.	ADV	adverb
        8.	AX	auxiliary verb (including copula)
        9.	AXD	auxiliary verb, past tense
        10.	CL	classifier
        11.	CONJ	coordinating conjunction
        12.	D	determiner
        13.	FN	formal noun
        14.	FW	foreign word
        15.	INTJ	interjection
        16.	MD	modal element
        17.	N	noun
        18.	NEG	negation
        19.	NPR	proper noun
        20.	NUM	numeral
        21.	P	particle
        22.	P-COMP	complementizer
        23.	P-CONN	conjunctional particle
        24.	P-FINAL	final particle
        25.	P-OPTR	operator
        26.	P-ROLE	role particle
        27.	PASS	passive - and there are PASS2
        28.	PNL	prenominal
        29.	PRO	pronoun
        30.	Q	quantifier
        31.	QN	noun with quantifier
        32.	SYM	symbol
        33.	VB	verb (or verb stem)
        34.	VB0	light verb
        35.	VB2	secondary verb
        36.	WADV	indeterminate adverb
        37.	WD	indeterminate determiner
        38.	WNUM	indeterminate numeral
        39.	WPRO	indeterminate pronoun
        40.	LS	List item marker (not officially claimed)'''), # https://npcmj.ninjal.ac.jp/wp-content/uploads/2019/05/npcmj_annotation_manual_en_201904.pdf
}

part_of_speech[C_NPCMJ] = part_of_speech[C_KTB]

# for corp, tags in part_of_speech.items():
#     print(corp)
#     print(' '.join(tags))