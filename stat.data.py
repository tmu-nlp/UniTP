#!/usr/bin/env python3
from os import error
from utils.types import M_TRAIN, M_DEVEL, M_TEST, E_ORIF5_HEAD, num_threads, E_ORIF4
from sys import argv
from collections import defaultdict, Counter
from tqdm import tqdm
from data.binary import xtype_to_logits, get_dir, get_rgt, get_jnt
from data.trapezoid.dataset import distribute_jobs, Queue, PennWorker, PennTreeKeeper, Tree
from data.multib.dataset import MAryWorker
from data.cross.multib import TreeKeeper, total_fence
from data.penn_types import CorpusReader, C_CTB, C_PTB, C_KTB
from data.stutt_types import C_DPTB, C_TIGER
from data.mp import mp_workers
from nltk.corpus import BracketParseCorpusReader
from data.io import load_i2vs, encapsulate_vocabs
from utils.str_ops import strange_to
from utils.file_io import join, isfile, isdir, create_join
from utils.yaml_io import load_yaml
from utils.shell_io import byte_style
from manager import _mfile, _lfile

csv_fpath = create_join('R_ggplot', 'stat.data')
F_MULTI = 'multi'

def write_complexity(complexity_gen, corp_name):
    with open(join(csv_fpath, f'sent.orif-node.{corp_name}.csv'), 'w') as fw:
        fw.write('len,height,orif,size\n')
        for line in complexity_gen():
            fw.write(line)

def write_compress_ratio(corp_name, ratios, append_multi = False):
    if append_multi: ratios = {F_MULTI: ratios}
    with open(join(csv_fpath, f'compress.ratio.{corp_name}.csv'), 'wa'[append_multi]) as fw:
        if not append_multi:
            fw.write('size,ratio,count,orif\n')
        for fct in ratios:
            for (last_size, this_size), cnt in ratios[fct].items():
                fw.write(f'{last_size},{this_size/last_size},{cnt},{fct}\n')

def write_lnr(corp_name, local_path, factors, extra_nlr = None, with_joint = False):
    with open(join(csv_fpath, f'orif-LNR.{corp_name}.csv'), 'w') as fw:
        if extra_nlr:
            assert not with_joint
            fw.write('orif,left,neutral,right,type\n')
            for fct, nlr in extra_nlr.items():
                fw.write(f"{fct},{nlr['<']},{nlr['-']},{nlr['>']},S1\n")
        else:
            line = 'orif,left,neutral,right'
            if with_joint:
                line += ',joint'
            fw.write(line + '\n')
        for fct in factors:
            triangle_nlr = defaultdict(int)
            with open(join(local_path, f'stat.xtype.{fct}')) as fr:
                for line in fr:
                    xtype, cnt = line.split()
                    cnt = int(cnt)
                    triangle_nlr[xtype[0]] += cnt
                    if 'j' in xtype:
                        triangle_nlr['j'] += cnt
            line = f"{fct},{triangle_nlr['<']},{triangle_nlr['-']},{triangle_nlr['>']}"
            if with_joint:
                line += f",{triangle_nlr['j']}\n"
            else:
                line += ',S2\n' if extra_nlr else '\n'
            fw.write(line)

def continuous_stratification(corp_name, local_path, corp_path):

    i2vs = load_i2vs(local_path, 'word tag label'.split())
    i2vs, field_v2is = encapsulate_vocabs(i2vs, {})

    _, w2i = field_v2is['word']
    _, t2i = field_v2is['tag']
    _, l2i = field_v2is['label']
    x2i = lambda x: xtype_to_logits(x, to_str = False)
    v2is = w2i, t2i, l2i, x2i
    
    if corp_name == C_PTB:
        reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")
        samples = []
        for fn in reader.fileids():
            # if 1 < int(fn[:2]) < 24:
            samples.extend(tree for tree in reader.parsed_sents(fn))
    else:
        reader = CorpusReader(corp_path)
        if corp_name == C_CTB:
            dataset = strange_to('001-270,271-300,301-325,440-1151', lambda x: f'{x:04}')
            in_dataset = lambda fpath: fpath[5:-3] in dataset
        else:
            reader.break_corpus(400)
        samples = []
        for fn in reader.fileids():
            if corp_name == C_KTB or in_dataset(fn):
                samples.extend(tree for tree in reader.parsed_sents(fn))

    fnames = reader.fileids()
    if corp_name == C_CTB:
        data_split = strange_to('001-325,440-1151', lambda x: f'{x:04}')
        fnames = [fn for fn in reader.fileids() if fn[5:-3] in data_split]
    b_q, m_q = Queue(), Queue()
    trapezoid_height = 1
    factors = E_ORIF4
    b_works = distribute_jobs(fnames, num_threads)
    m_works = distribute_jobs(samples, num_threads)
    for i in range(num_threads):
        w = PennWorker(b_q, reader, b_works[i], trapezoid_height, v2is, factors, corp_name == C_KTB)
        w.start()
        b_works[i] = w
        w = MAryWorker(m_q, m_works[i], v2is[:3], False, corp_name == C_KTB)
        w.start()
        m_works[i] = w
    def b_core_fn(words, length, tree_str, factored):
        keeper = PennTreeKeeper(Tree.fromstring(tree_str), v2is, trapezoid_height)
        keeper.update_factored(factored, words)
        return (keeper,)
    def m_core_fn(words, signals, info):
        return (signals,)
    print(f'With {num_threads} threads for {corp_name}:')
    keepers, = mp_workers(b_works, b_q, b_core_fn, 1, '  B-S2')
    samples, = mp_workers(m_works, m_q, m_core_fn, 1, '  M')

    nlr = {fct: defaultdict(int) for fct in factors}
    ratios = {fct: defaultdict(int) for fct in factors + (F_MULTI,)}
    def complexity_gen():
        for keeper in keepers:
            for fct in factors:
                token_size = len(keeper[fct]['token'])
                label_size = 0
                height = len(keeper[fct]['label'])
                for x in keeper[fct]['label']:
                    this_size = len(x)
                    if label_size:
                        ratios[fct][(last_size, this_size)] += 1
                    label_size += this_size
                    last_size = this_size
                yield f'{token_size},{height},{fct},{label_size}\n'
                fct_nlr = nlr[fct]
                for xtype in keeper[fct]['xtype']:
                    
                    for direc, right in zip(get_dir(xtype), get_rgt(xtype)):
                        if direc:
                            fct_nlr['<>'[int(right)]] += 1
                        else:
                            fct_nlr['-'] += 1
                    for joint in get_jnt(xtype):
                        if joint:
                            fct_nlr['j'] += 1

        for signals in samples:
            w, t, ll, lc = signals
            layer_size = 0
            height = len(ll)
            for x in ll:
                this_size = len(x)
                if layer_size:
                    ratios[F_MULTI][(last_size, this_size)] += 1
                last_size = this_size
                layer_size += this_size
            yield f'{len(w)},{height},{F_MULTI},{layer_size}\n'

    write_complexity(complexity_gen, corp_name)
    write_compress_ratio(corp_name, ratios)
    write_lnr(corp_name, local_path, factors, nlr)

def disco_orientation(corp_name, local_path, factors):
    ratios = {fct: defaultdict(int) for fct in factors}
    def complexity_gen():
        with tqdm(desc = corp_name) as qbar:
            first_run = True
            for factor in factors:
                for mode in (M_TRAIN, M_DEVEL, M_TEST):
                    with open(join(local_path, f'{mode}.index.{factor}')) as fr:
                        for _, line in enumerate(fr):
                            lengths = tuple(int(x) for x in line.split())
                            height = len(lengths)
                            if not lengths:
                                finish = True
                                continue
                            for sizes in zip(lengths, lengths[1:]):
                                ratios[factor][sizes] += 1
                            yield f'{lengths[0]},{height},{factor},{sum(lengths)}\n'
                            qbar.update(1)
                        assert finish
                if first_run:
                    qbar.total = qbar.n * len(factors)
                    first_run = False
            qbar.total = qbar.n #???

    write_complexity(complexity_gen, corp_name)
    write_compress_ratio(corp_name, ratios)
    write_lnr(corp_name, local_path, factors, with_joint = True)

def live_dm_layer_stat(*keeper_factor):
    if not keeper_factor:
        sent_head = 'len,height,gap,linear,square_layers,square,max_square,medf\n'
        layer_head = 'len,height,ratio,n_comp,n_child,n_positive,n_chunk,medf\n'
        return sent_head, layer_head
    keeper, factor = keeper_factor
    layers_of_label, layers_of_space, layers_of_disco = keeper.stratify(factor)
    lid = -1
    layer_lines = ''
    ratio_count = defaultdict(int)
    for lid, (space_layer, disco_layer, next_label_layer) in enumerate(zip(layers_of_space, layers_of_disco, layers_of_label[1:])):
        next_len = len(next_label_layer)
        this_len, chunk_layer = total_fence(space_layer)
        ratio_count[(this_len, next_len)] += 1
        n_comp = len(disco_layer)
        n_children = sum(len(g) for g in disco_layer.values())
        n_positive = sum(len(g)**2 for g in disco_layer.values())
        layer_lines += f'{this_len},{lid},{next_len/this_len},{n_comp},{n_children},{n_positive},{len(chunk_layer)},{factor}\n'

    linear = sum(len(x) for x in layers_of_label)
    square_layers = sum(1 for gs in layers_of_disco if gs)
    square = sum(sum(len(g) for g in gs.values())**2 for gs in layers_of_disco)
    max_square = sum(len(x)**2 for x in layers_of_label)
    sent_line = f'{len(keeper.text)},{lid+1},{keeper.gaps},{linear},{square_layers},{square},{max_square},{factor}\n'
    orif_line = f'{len(keeper.text)},{len(layers_of_label)},{F_MULTI},{linear}\n'
    return sent_line, layer_lines, orif_line, ratio_count

def disco_multib(corp_name, data_specs, factors):
    from data.stutt_types import select_and_split_corpus
    corp_spec = data_spec[corp_name]
    build_params = corp_spec['build_params']
    (corpus, _, _, _,
     read_func) = select_and_split_corpus(corp_name,
                                          corp_spec['source_path'],
                                          build_params['train_set'],
                                          build_params['devel_set'],
                                          build_params['test_set'],
                                          binary = False, read_dep = join(base_path, 'data'))
    ratio_count = Counter()
    errors = []
    with open(join(csv_fpath, f'sent.medf-node.{corp_name}.csv'), 'w') as fws,\
         open(join(csv_fpath, f'sent.orif-node.{corp_name}.csv'), 'a') as fas,\
         open(join(csv_fpath, f'compress.ratio.matrix.{corp_name}.csv'), 'w') as fwl,\
         open(join(csv_fpath, f'error.head.{corp_name}.csv'), 'w') as fwe:
        sh, lh = live_dm_layer_stat()
        fws.write(sh)
        fwl.write(lh)
        for tid, tree, dep in tqdm(corpus, desc = corp_name):
            try:
                keeper = read_func(tree, dep = dep, verbose_file = (tid, fwe))#, details = True)
            except:
                errors.append(tid)
                continue
            if keeper.has_signals:
                for fct in factors:
                    sl, ll, ol, rc = live_dm_layer_stat(keeper, fct)
                    fws.write(sl)
                    fwl.write(ll)
                fas.write(ol)
                ratio_count += rc
    write_compress_ratio(corp_name, ratio_count, True)
    if errors: print('  with errors:', ', '.join(errors))

def discontinuous_stratification(corp_name, local_path, source_path):
    from data.cross.multib import E_FACTOR
    disco_orientation(corp_name, local_path, E_ORIF5_HEAD)
    disco_multib(corp_name, source_path, E_FACTOR)

_, base_path = argv
mfile = join(base_path, _mfile)
lfile = join(base_path, _lfile)
data_spec = load_yaml(mfile, lfile, wait = False)['data']


def run(corpura, func):
    for corp_name in corpura:
        source_path = data_spec[corp_name]['source_path']
        local_path = join(base_path, 'data', corp_name)

        if not isdir(local_path):
            print(byte_style(corp_name + f' is not prepared: use ./manager {base_path} -p', '1'))
            continue
        if not isdir(source_path) and (corp_name != C_TIGER or not isfile(source_path)):
            print(byte_style(corp_name + f' source_path is invalid: check key under data.{corp_name} in {mfile}', '1'))
            continue
        func(corp_name, local_path, source_path)

run((C_PTB, C_CTB, C_KTB), lambda *args: continuous_stratification(*args))
run((C_DPTB, C_TIGER), lambda *args: discontinuous_stratification(*args))