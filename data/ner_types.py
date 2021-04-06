def read_dataset(fname):
    words = []
    pos_tags = []
    ner_tags = []
    with open(fname) as fr:
        for line in fr:
            if line == '\n':
                yield (words, pos_tags, ner_tags)
                words = []
                pos_tags = []
                ner_tags = []
                continue
            word, pos, ner = line.split()
            words.append(word)
            pos_tags.append(pos)
            ner_tags.append(ner)

from random import random
def remove_bio_prefix(bio_tags):
    ner_tags, ner_fence = [], []
    last_ner = None
    for nid, ner in enumerate(bio_tags):
        is_bi_tag = '-' in ner
        if is_bi_tag:
            prefix, ner = ner.split('-')
            if prefix == 'B':
                last_ner = None
        if ner != last_ner:
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
C_ABSTRACT = 'ner'
E_NER = C_IN

build_params = {C_IN: {}}
ft_bin = {C_IN: 'id'}

from data.io import make_call_fasttext, check_fasttext
call_fasttext = make_call_fasttext(ft_bin)


from data.stan_types import split_files, M_TRAIN, M_DEVEL, M_TEST, NIL # same name
from os.path import join, isfile
from collections import Counter, defaultdict
from data.io import check_vocab, save_vocab
from utils.pickle_io import pickle_dump, pickle_load
from tqdm import tqdm
def build(save_to_dir,
          corp_path,
          corp_name,
          **kwargs):

    tok_cnt = Counter()
    pos_cnt = Counter()
    bio_cnt = Counter()
    ner_cnt = Counter()
    ds_len_cnts = {}
    ds_ner_cnts = {}
    desc = ''
    with tqdm(split_files.items()) as qbar:
        for src, dst in qbar:
            desc += dst
            ds_len_cnt = defaultdict(int)
            ds_ner_cnt = defaultdict(int)
            for sid, (wd, pt, nt) in enumerate(read_dataset(join(corp_path, src + '.txt'))):
                tok_cnt += Counter(wd)
                pos_cnt += Counter(pt)
                bio_cnt += Counter(nt)
                _, nt = remove_bio_prefix(nt)
                ner_cnt += Counter(nt)
                ds_len_cnt[len(wd)] += 1
                ds_ner_cnt[sum(1 for x in nt if x != 'O')] += 1
                qbar.desc = 'Loading ' + desc + f'({sid})'
                # TODO should we use nil?
            ds_len_cnts[dst] = ds_len_cnt
            ds_ner_cnts[dst] = ds_ner_cnt
            desc += f'({sid}); '
        qbar.desc = 'Loaded ' + desc
    pickle_dump(join(save_to_dir, 'info.pkl'), (ds_len_cnts, ds_ner_cnts))
    chr_cnt = defaultdict(int)
    for tk, cnt in tok_cnt.items():
        for ch in tk:
            chr_cnt[ch] += cnt
    tok_file = join(save_to_dir, 'vocab.word')
    chr_file = join(save_to_dir, 'vocab.char')
    pos_file = join(save_to_dir, 'vocab.pos')
    ner_file = join(save_to_dir, 'vocab.ner')
    bio_file = join(save_to_dir, 'vocab.bio')
    _, ts = save_vocab(tok_file, tok_cnt, [NIL])
    _, cs = save_vocab(chr_file, chr_cnt, [NIL])
    _, ps = save_vocab(pos_file, pos_cnt, [NIL])
    _, ns = save_vocab(ner_file, ner_cnt, [NIL])
    _, bs = save_vocab(bio_file, bio_cnt, [NIL])
    return ts, cs, ps, ns, bs

def check_data(save_dir, valid_sizes):
    from sys import stderr
    from utils.str_ops import histo_count
    try:
        tok_size, chr_size, pos_size, ner_size, bio_size = valid_sizes
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    fname = join(save_dir, 'info.pkl')
    if not isfile(fname):
        print('Not found: info.pkl')
        return False
    tok_file = join(save_dir, 'vocab.word')
    chr_file = join(save_dir, 'vocab.char')
    pos_file = join(save_dir, 'vocab.pos')
    ner_file = join(save_dir, 'vocab.ner')
    bio_file = join(save_dir, 'vocab.bio')
    res = check_vocab(tok_file, tok_size)
    res = res and check_vocab(chr_file, chr_size)
    res = res and check_vocab(pos_file, pos_size)
    res = res and check_vocab(ner_file, ner_size)
    res = res and check_vocab(bio_file, bio_size)
    ds_len_cnts, ds_ner_cnts = pickle_load(fname)
    print(f'Length distribution in [ Train set ] ({sum(ds_len_cnts[M_TRAIN].values())})', file = stderr)
    print(histo_count(ds_len_cnts[M_TRAIN], bin_size = 10), file = stderr)
    print('  #NER per sentence dist.:', file = stderr)
    print(histo_count(ds_ner_cnts[M_TRAIN], bin_size = 3), file = stderr)
    print(f'Length distribution in [ Dev set ]   ({sum(ds_len_cnts[M_DEVEL].values())})', file = stderr)
    print(histo_count(ds_len_cnts[M_DEVEL], bin_size = 10), file = stderr)
    print('  #NER per sentence dist.:', file = stderr)
    print(histo_count(ds_ner_cnts[M_DEVEL], bin_size = 3), file = stderr)
    print(f'Length distribution in [ Test set ]  ({sum(ds_len_cnts[M_TEST].values())})', file = stderr)
    print(histo_count(ds_len_cnts[M_TEST], bin_size = 10), file = stderr)
    print('  #NER per sentence dist.:', file = stderr)
    print(histo_count(ds_ner_cnts[M_TEST], bin_size = 3), file = stderr)
    return res

from nltk.tree import Tree
def bio_to_tree(words, bio_tags, pos_tags = None, root_label = 'TOP'): # TODO perserve bio
    bottom = []
    last_ner = last_chunk = None
    for nid, (word, bio) in enumerate(zip(words, bio_tags)):
        leaf = Tree(f'#{nid+1}' if pos_tags is None else pos_tags[nid], [word])
        if '-' in bio:
            prefix, ner = bio.split('-')
        else:
            prefix = bio
            ner = None
        # import pdb; pdb.set_trace()
        if last_ner == ner:
            if ner is None:
                bottom.append(leaf)
                last_chunk = None
            else:
                last_chunk.append(leaf)
        else: # changed
            if last_chunk:
                bottom.append(Tree(last_ner, last_chunk))
            if ner is None:
                bottom.append(leaf)
                last_chunk = None
            else:
                last_chunk = [leaf]
            last_ner = ner
    if last_chunk:
        bottom.append(Tree(last_ner, last_chunk))
    return Tree(root_label, bottom)

def ner_to_tree(words, ner_tags, fences, pos_tags = None, perserve_fence = False, weights = None, root_label = 'TOP'):
    leaves = []
    for nid, word in enumerate(words):
        if pos_tags is None:
            if weights is None:
                label = f'#{nid+1}'
            else:
                label = f'{weights[nid] * 100:.0f}%'
            leaf = Tree(label, [word])    
        else:
            leaf = Tree(pos_tags[nid], [word])
            if weights is not None:
                leaf = Tree(f'{weights[nid] * 100:.0f}%', [leaf])
        leaves.append(leaf)
    bottom = []
    # import pdb; pdb.set_trace()
    for ner, start, end in zip(ner_tags, fences, fences[1:]):
        if ner == 'O' and not perserve_fence:
            bottom.extend(leaves[start: end])
        else:
            bottom.append(Tree(ner, leaves[start: end]))
    return Tree(root_label, bottom)