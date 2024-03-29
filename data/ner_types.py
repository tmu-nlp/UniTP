def read_dataset(fname, n):
    sent = tuple([] for _ in range(n))
    with open(fname) as fr:
        for line in fr:
            if line.startswith('-DOCSTART-'):
                assert next(fr) == '\n'
                continue
            if line == '\n':
                yield sent
                sent = tuple([] for _ in range(n))
                continue
            for col, val in zip(sent, line.split()):
                col.append(val)

def has_o_structure(nid, chunks):
    if nid == 0 or not chunks:
        return False
    lhs = chunks[nid - 1] # O I-NP I-NP B-VP
    rhs = chunks[nid]
    if lhs != rhs:
        if lhs[:2] == 'B-' and lhs[:2] == rhs[:2]:
            return False
        return True
    return False
    

from random import random
def remove_bio_prefix(bio_tags, chunks = None):
    ner_tags, ner_fence = [], []
    last_ner = None
    for nid, ner in enumerate(bio_tags):
        if '-' in ner:
            prefix, ner = ner.split('-')
            if prefix == 'B':
                last_ner = None
        if ner != last_ner or ner == 'O' and has_o_structure(nid, chunks):
            ner_fence.append(nid)
            ner_tags.append(ner)
            last_ner = ner
    ner_fence.append(len(bio_tags))
    return ner_fence, ner_tags

def recover_bio_prefix(chunk, ner_tags):
    long_ner_tags = []
    for start, end, ner in zip(chunk, chunk[1:], ner_tags):
        if ner == 'O':
            children = ['O'] * (end - start)
        else:
            children = [('I-' if i else 'B-') + ner for i in range(end - start)]
        long_ner_tags += children
    return long_ner_tags

C_IN = 'idner'
C_EN = 'conll'
E_NER = C_IN, C_EN

from utils.types import true_type, false_type, rate_5, K_CORP

augment = dict(a = rate_5, p = rate_5)
ner_extension  = dict(break_o_chunk = rate_5,
                      break_whole   = false_type,
                      delete        = augment,
                      insert        = augment,
                      substitute    = augment)
def fill_accp_data_config(accp_data_config):
    accp_data_config['ner'] = dict(with_bi_prefix = true_type,
                                   with_pos_tag   = true_type)
    corp_configs = {
        C_IN: dict(extension = ner_extension), 
        C_EN: dict(extension = dict(with_o_chunk = true_type))
    }
    accp_data_config[K_CORP].update(corp_configs)


from data.io import make_call_fasttext, check_fasttext, check_vocab, save_vocab, split_dict
from data.stan_types import sstb_split # same name
build_params = {C_IN: sstb_split, C_EN: split_dict('eng.train', 'eng.testa', 'eng.testb')}
ft_bin = {C_IN: 'id', C_EN: 'en'}
call_fasttext = make_call_fasttext(ft_bin)

from os.path import join
from collections import Counter, defaultdict
from utils.str_ops import StringProgressBar
from data import NIL

def build(save_to_dir,
          corp_path,
          corp_name,
          **split_files):

    tok_cnt = Counter()
    pos_cnt = Counter()
    bio_cnt = Counter()
    ner_cnt = Counter()
    ds_len_cnts = {}
    ds_ner_cnts = {}
    n = {C_IN: 3, C_EN: 4}[corp_name]
    names = tuple(split_files.keys())
    with StringProgressBar(names, sep = ', ', prefix = 'Load ') as qbar:
        for dst, src in split_files.items():
            ds_len_cnt = defaultdict(int)
            ds_ner_cnt = Counter()
            for sid, largs in enumerate(read_dataset(join(corp_path, src), n)):
                if n == 3:
                    wd, pt, nt = largs
                else:
                    wd, pt, _, nt = largs
                tok_cnt += Counter(wd)
                pos_cnt += Counter(pt)
                bio_cnt += Counter(nt)
                _, nt = remove_bio_prefix(nt)
                ner_cnt += Counter(nt)
                ds_len_cnt[len(wd)] += 1
                ds_ner_cnt.update(t for t in nt if t != 'O')
            ds_len_cnts[dst] = ds_len_cnt
            ds_ner_cnts[dst] = ds_ner_cnt
            qbar.update(names.index(dst), total = 1)
            qbar.update(names.index(dst))
            prefix, suffix = qbar.desc
            qbar.desc = prefix, (suffix[:-1] + f', {sid})') if suffix else (f' ({sid})')
    tok_file = join(save_to_dir, 'vocab.word')
    pos_file = join(save_to_dir, 'vocab.pos')
    ner_file = join(save_to_dir, 'vocab.ner')
    bio_file = join(save_to_dir, 'vocab.bio')
    ts = save_vocab(tok_file, tok_cnt, [NIL])
    ps = save_vocab(pos_file, pos_cnt, [NIL])
    ns = save_vocab(ner_file, ner_cnt, [NIL])
    bs = save_vocab(bio_file, bio_cnt, [NIL])

    from utils.str_ops import histo_count
    with open(join(save_to_dir, 'vocab.rpt'), 'w') as fw:
        for ds, lc in ds_len_cnts.items():
            fw.write(f'Length distribution in [ {ds.upper()} set ] ({sum(lc.values())})\n')
            fw.write(histo_count(lc, bin_size = 10) + '\n')
        for ds, nc in ds_ner_cnts.items():
            fw.write(f'NER frequency in [ {ds.upper()} set ] ({sum(nc.values())})\n')
            fw.write(', '.join(n + f': {c}' for n, c in nc.items()) + '.\n')
    return ts, ps, ns, bs

def check_data(save_dir, valid_sizes):
    from sys import stderr
    from utils.str_ops import histo_count
    try:
        tok_size, pos_size, ner_size, bio_size = valid_sizes
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    tok_file = join(save_dir, 'vocab.word')
    pos_file = join(save_dir, 'vocab.pos')
    ner_file = join(save_dir, 'vocab.ner')
    bio_file = join(save_dir, 'vocab.bio')
    res = check_vocab(tok_file, tok_size)
    res = res and check_vocab(pos_file, pos_size)
    res = res and check_vocab(ner_file, ner_size)
    res = res and check_vocab(bio_file, bio_size)
    return res

from random import random
def break_o(old_ner, old_fence, break_o_chunk, break_whole, o_idx):
    # 0359 -> 012356789
    #  OPO ->  OOOPOOOO
    new_ner, new_fence = [], [0]
    for ner, start, end in zip(old_ner, old_fence, old_fence[1:]):
        if ner == o_idx and end - start > 1:
            new_range = range(start + 1, end + 1)
            if not break_whole:
                new_range = [i for i in new_range if i == end or random() < break_o_chunk]
            new_ner.extend(ner for _ in new_range)
            new_fence.extend(new_range)
        else:
            new_ner.append(ner)
            new_fence.append(end)
    return new_ner, new_fence

def insert_o(old_ner, old_fence, f_indices, o_idx):
    # 01234567 -> 01_34567_9
    # 0358 -> 02346X
    #  OPO ->  OXOPOXO
    # assert len(f_indices) == len(values)
    assert all(x <= y for x, y in zip(f_indices, f_indices[1:]))
    insersion = 0
    new_ner, new_fence = [], [0]
    for ner, start, end in zip(old_ner, old_fence, old_fence[1:]):
        inserted = None
        while insersion < len(f_indices) and (ptr := f_indices[insersion]) <= end: # py38
            # print(ner, insersion, 'WHILE', new_fence, new_ner)
            # import pdb; pdb.set_trace()
            if (lhs := ptr + insersion) != new_fence[-1]:
                new_fence.append(lhs)
                if ptr != inserted:
                    new_ner.append(ner)
            insersion += 1
            new_ner.append(o_idx)
            if (rhs := lhs + 1) < end + insersion:
                new_fence.append(rhs)
            inserted = ptr
        else:
            # print(ner, insersion, 'ELSE', new_fence, end + insersion, new_ner)
            # import pdb; pdb.set_trace()
            if inserted != end:
                new_ner.append(ner)
            new_fence.append(end + insersion)
    return new_ner, new_fence

def delete_o(old_ner, old_fence, n_indices):
    deletion = 0
    new_ner, new_fence = [], [0]
    assert all(x < y for x, y in zip(n_indices, n_indices[1:]))
    for ner, start, end in zip(old_ner, old_fence, old_fence[1:]):
        offset = 0
        deleted = None
        while (has_idx := deletion < len(n_indices)) and (next_ptr := n_indices[deletion]) == start + offset and next_ptr < end:
            offset += 1
            deleted = next_ptr
            deletion += 1
        if deleted is not None and deleted == end - 1: # deleted whole
            continue
        elif has_idx and next_ptr < end: # rhs
            while next_ptr < end: # discontinuous chunks
                if new_fence[-1] < next_ptr - deletion:
                    new_ner.append(ner)
                    new_fence.append(next_ptr - deletion)
                deletion += 1
                if deletion >= len(n_indices):
                    break # else cannot catch this
                next_ptr = n_indices[deletion]
            if next_ptr < end - 1: # remaining
                new_ner.append(ner)
                new_fence.append(end - deletion)
        else: # no next_ptr < end, no delete
            new_ner.append(ner)
            new_fence.append(end - deletion)
    return new_ner, new_fence
    
def substitute_o(old_ner, old_fence, n_indices, o_idx):
    substution = 0
    new_ner, new_fence = [], [0]
    for ner, start, end in zip(old_ner, old_fence, old_fence[1:]):
        substituted = None
        # import pdb; pdb.set_trace()
        while substution < len(n_indices) and (lhs := n_indices[substution]) < end:
            if new_fence[-1] < lhs:
                new_ner.append(ner)
                new_fence.append(lhs)
            new_ner.append(o_idx)
            if (rhs := lhs + 1) < end:
                new_fence.append(rhs)
            substution += 1
            substituted = lhs
        if substituted is None or substituted < end - 1:
            new_ner.append(ner)
        new_fence.append(end)
    return new_ner, new_fence

from nltk.tree import Tree
from data import brackets2RB
def bio_to_tree(words, bio_tags, pos_tags = None, show_internal = False, root_label = 'TOP'): # TODO perserve bio
    bottom = []
    last_ner = last_chunk = None
    prefixes = ''
    for nid, (word, bio) in enumerate(zip(words, bio_tags)):
        leaf = Tree(f'#{nid+1}' if pos_tags is None else brackets2RB(pos_tags[nid]), [brackets2RB(word, True)])
        prefix, ner = bio.split('-') if '-' in bio else (bio, None)
        if last_ner == ner and prefix != 'B':
            if ner is None:
                bottom.append(leaf)
                last_chunk = None
            else:
                if show_internal: prefixes += prefix
                last_chunk.append(leaf)
        else: # changed
            if last_chunk:
                if show_internal: prefixes = '.' + prefixes
                bottom.append(Tree(last_ner + prefixes, last_chunk))
            if ner is None:
                bottom.append(leaf)
                last_chunk = None
            else:
                if show_internal: prefixes = prefix
                last_chunk = [leaf]
            last_ner = ner
    if last_chunk:
        bottom.append(Tree(last_ner, last_chunk))
    return Tree(root_label, bottom)

def ner_to_tree(words, ner_tags, fences, pos_tags = None, show_internal = False, weights = None, root_label = 'TOP'):
    leaves = [Tree(f'#{nid+1}' if pos_tags is None else brackets2RB(pos_tags[nid]), [brackets2RB(word, True)]) for nid, word in enumerate(words)]
    bottom = []
    all_os = all(x == 'O' for x in ner_tags)
    for ner, start, end in zip(ner_tags, fences, fences[1:]):
        children = leaves[start:end]
        if end - start > 1 and weights is not None:
            children = [Tree(f'{weight * 100:.0f}%', [leaf]) for leaf, weight in zip(children, weights[start:end])]
        if ner == 'O' and (not show_internal or all_os or (end - start == 1)):
            bottom.extend(children)
        else:
            bottom.append(Tree(ner, children))
    return Tree(root_label, bottom)