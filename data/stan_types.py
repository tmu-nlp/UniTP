C_SSTB = 'sstb'

build_params = {C_SSTB: {}}
ft_bin = {C_SSTB: 'en'}

from data.io import make_call_fasttext, check_fasttext
from utils.types import M_TRAIN, M_TEST, M_DEVEL, UNK, NIL
call_fasttext = make_call_fasttext(ft_bin)

__datasets__ = {'train': M_TRAIN, 'dev': M_DEVEL, 'test': M_TEST}

from nltk.tree import Tree
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.pickle_io import pickle_dump, pickle_load
from utils.str_ops import histo_count
from os.path import join, isfile
from data.io import check_vocab, save_vocab, sort_count
from sys import stderr
from data.delta import DeltaX, lnr_order, xtype_to_logits

def string_to_pol_word_syn(line):
    tree = Tree.fromstring(line)
    x    = DeltaX.from_stan(tree)
    polar, direc, _ = x.to_triangles()
    words = tree.leaves()
    polar = tuple(p[0] if isinstance(p, list) else p for p in polar)
    return words, polar, direc

def _build_one(from_file, syn_file, word_file, xty_file):
    word_cnt = Counter()
    syn_cnt  = Counter()
    xty_cnt  = Counter()
    len_cnt  = defaultdict(int)
    with open(from_file, 'rb') as fr,\
         open(syn_file,  'w', encoding = 'utf-8') as fs,\
         open(word_file, 'w', encoding = 'utf-8') as fw,\
         open(xty_file,  'w', encoding = 'utf-8') as fd:
        for line in tqdm(fr, desc = from_file):
            line = line.replace(b'\\/', b'/').replace(b'\xc2\xa0', b'.').decode('utf-8')
            words, polar, direc = string_to_pol_word_syn(line) # terrible sign
            word_cnt += Counter(words)
            syn_cnt  += Counter(polar)
            xty_cnt  += Counter(direc)
            direc = tuple(xtype_to_logits(x) for x in direc)
            fw.write(' '.join(words) + '\n')
            fs.write(' '.join(polar) + '\n')
            fd.write(' '.join(direc) + '\n')
            len_cnt[len(words)] += 1
    return word_cnt, syn_cnt, xty_cnt, len_cnt

def build(save_to_dir, stree_path, corp_name, verbose = True, **kwargs):
    assert corp_name == 'sstb'

    info = {}
    tok_cnt, syn_cnt, xty_cnt = Counter(), Counter(), Counter()
    for src, dst in __datasets__.items():
        from_file = join(stree_path,  f'{src}.txt')
        word_file = join(save_to_dir, f'{dst}.word')
        syn_file  = join(save_to_dir, f'{dst}.polar')
        xty_file  = join(save_to_dir, f'{dst}.xtype')
        x = _build_one(from_file, syn_file, word_file, xty_file)
        if dst == M_TRAIN:
            train_cnt = x[0]
        tok_cnt += x[0]
        syn_cnt += x[1]
        xty_cnt += x[2]
        info[dst] = x[3]
        print(f'Length distribution in [ {dst.title()} set ]', file = stderr)
        print(histo_count(x[3], bin_size = 10), file = stderr)

    pickle_dump(join(save_to_dir, 'info.pkl'), info)

    syn_file = join(save_to_dir, 'vocab.polar')
    tok_file = join(save_to_dir, 'vocab.word')
    xty_file = join(save_to_dir, 'vocab.xtype')
    _, ts = save_vocab(tok_file, tok_cnt, [NIL, UNK] + sort_count(train_cnt))
    _, ss = save_vocab(syn_file, syn_cnt, [NIL     ])
    _, dr = save_vocab(xty_file, xty_cnt, lnr_order(xty_cnt)[0])
    return len(train_cnt), ts, ss, dr

def check_data(save_dir, valid_sizes):
    try:
        _, tok_size, syn_size, xty_size = valid_sizes
    except:
        print('Should check vocab with compatible sizes, even Nones', file = stderr)
        return False
    tok_file = join(save_dir, 'vocab.word')
    syn_file = join(save_dir, 'vocab.polar')
    xty_file = join(save_dir, 'vocab.xtype')
    res = check_vocab(tok_file, tok_size)
    res = res and check_vocab(syn_file, syn_size)
    res = res and check_vocab(xty_file, xty_size)

    fname = join(save_dir, 'info.pkl')
    if res and isfile(fname):
        info = pickle_load(fname)
        print('Total:', sum(sum(info[ds].values()) for ds in __datasets__.values()), file = stderr)
        for ds in __datasets__.values():
            print(f'Length distribution in [ {ds.title()} set ] ({sum(info[ds].values())})', file = stderr)
            print(histo_count(info[ds], bin_size = 10), file = stderr)
    
    return res