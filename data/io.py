from collections import defaultdict, Counter, namedtuple
import sys
from os.path import isfile, join, basename
from data.delta import LogitX


from collections import namedtuple
TreeSpecs = namedtuple('TreeSpecs', 'word_size, tag_size, label_size, ftag_size, pre_trained')
TreeBatch = namedtuple('TreeBatch', f'initializer, reinit_rand, word, tag, label, {",".join("ori_" + x for x in LogitX._fields)}, ftag, finc, seq_len, num_sample, seq_size, batch_size, full_batch_size')
PyFields = namedtuple('PyFields', 'word, tag, label, ftag')

__fmt_cnt = '\t%d'

def load_i2vs(vocab_dir, suffixes):
    i2vs = {}
    for suf in suffixes:
        py_v = list(gen_vocab(join(vocab_dir, f'vocab.{suf}')))
        i2vs[suf] = py_v
    return i2vs

def encapsulate_vocabs(i2vs, oovs):
    def inner(i2v, f): # python bug: namespace
        size = len(i2v)
        v2i = {v:i for i,v in enumerate(i2v)}
        
        if f in oovs: # replace the dict function
            oov = oovs[f]
            v2i_func = lambda v: v2i.get(v, oov)
            assert oov in range(size)
        else:
            v2i_func = v2i.get
        return size, v2i_func

    v2is  = {}
    for f, i2v in i2vs.items():
        v2is[f] = inner(i2v, f)
        i2vs[f] = tuple(i2v)

    return i2vs, v2is

def extend_vocabs_joint(vocabs, main_tv, categorical_label):
    tv, sv = vocabs
    if main_tv:
        extra_tv = ((i, t) for i,t in enumerate(tv) if t not in main_tv) # exclude <nil>
        extra_id, extra_tv = zip(*extra_tv)
        assert UNK not in extra_tv
        tv = main_tv + extra_tv
    else:
        extra_id = None
    cs = categorical_label
    if categorical_label:
        sv = sv[:3] + tuple(sorted(sv[3:]))
    else:
        sv = sorted(sv[3:])
    assert ''.join(sv[-5:]) == '01234'
    tn = len(tv)
    sn = len(sv)
    tok_vocab = index_table_from_tensor(tf_tv, default_value = 0)
    pol_vocab = index_table_from_tensor(tf_sv, default_value = -1 if cs else 2)
    # tok_inv = tf.contrib.lookup.index_to_string_table_from_tensor(tf_tv, default_value = unk)
    # label_inv = tf.contrib.lookup.index_to_string_table_from_tensor(tf_sv, default_value = unk)
    # xty_inv = tf.contrib.lookup.index_to_string_table_from_tensor(tf_dv, default_value = unk)
    # (tok_inv, None, label_inv, xty_inv), 
    py_sizes  = tn, cs, sn, cs
    py_vocabs = tv, cs, sv, cs
    tf_tables = tok_vocab, cs, pol_vocab, cs
    return PyVocabs(*py_sizes), PyVocabs(*py_vocabs), PyVocabs(*tf_tables), extra_id

def make_call_fasttext(corp_ft_bin):
    from utils.shell_io import call_fasttext
    def inner(fasttext, path, corp_name):
        wfile = join(path, 'vocab.word')
        vfile = join(path, 'word.vec')
        ft_bin = corp_ft_bin[corp_name]
        ft_bin = fasttext['ft_bin'][ft_bin]
        print(f"calling fasttext for [{wfile}:{vfile}] of {corp_name} with '{basename(ft_bin)}'", file = sys.stderr)
        call_fasttext(fasttext['path'], wfile, vfile, ft_bin, fasttext['ft_lower'])
    return inner

def check_fasttext(path):
    wfile = join(path, 'word.vec')
    vfile = join(path, 'vocab.word')
    if isfile(wfile) and isfile(vfile):
        # import pdb; pdb.set_trace()
        from utils.file_io import count_lines
        wlen = count_lines(wfile, True) - 1
        vlen = count_lines(vfile, True)
        if wlen == vlen:
            return True
        else:
            print(f'not match', wlen, vlen)
    else:
        print('wvfiles not exist')
    return False

def get_fasttext(fname):
    import numpy as np
    return np.loadtxt(fname, dtype = np.float32)

def sort_count(cnt):
    return sorted(cnt.items(), key = lambda x:x[1], reverse = True)

def save_vocab(save_to_file, common_cnt, special_toks = [], appendix_cnt = None):
    # [NIL PAD] + [syn] + [pos]
    lines = []
    if appendix_cnt:
        appendix_cnt = appendix_cnt.copy()
    for stok in special_toks:
        if isinstance(stok, tuple):
            stok = stok[0]
        cnt = appendix_cnt.pop(stok) if appendix_cnt and stok in appendix_cnt else 0
        if stok in common_cnt:
            lines.append(stok + __fmt_cnt % (cnt + common_cnt.pop(stok)))
        else:
            lines.append(stok)

    if appendix_cnt:
        for tok in appendix_cnt:
            if tok in common_cnt:
                appendix_cnt[tok] += common_cnt.pop(tok)

    vocab = sort_count(common_cnt)
    if appendix_cnt:
        vocab += sort_count(appendix_cnt)

    for v,c in vocab:
        lines.append( v + __fmt_cnt % c )
    with open(save_to_file, 'w', encoding = 'utf-8') as fw:
        fw.write('\n'.join(lines))
    if appendix_cnt is not None:
        return len(special_toks), len(special_toks) + len(common_cnt), len(lines)
    return len(special_toks), len(lines)

def gen_vocab(fname):
    special_bound = None
    _delim = __fmt_cnt[0]
    with open(fname, 'r') as fr:
        for idx, tok in enumerate(fr):
            tok = tok.rstrip()
            if special_bound:
                tok = tok[:tok.find(_delim)]
            elif _delim in tok:
                special_bound = idx
                tok = tok[:tok.find(_delim)]
            yield tok

def check_vocab(fname, expected_size = None):
    print('check', fname, file = sys.stderr)
    special_bound = None
    _delim = __fmt_cnt[0]
    try:
        with open(fname, 'r') as fr:
            for idx, tok in enumerate(fr):
                if special_bound is not None:
                    if _delim not in tok:
                        print('Ill-formed', fname, file = sys.stderr)
                        return False
                elif _delim in tok:
                    special_bound = idx
    except Exception as e:
        print(e, file = sys.stderr)
        return False
    if expected_size and expected_size != idx + 1:
        print('Invalid size %d vs. %d' % (expected_size, idx + 1), fname, file = sys.stderr)
        return False
    return True

def exam_trace_mark(pickle_to = None):
    from nltk.corpus import treebank as ptb
    unary_trans = defaultdict(Counter)
    vocab = Counter()
    for i, tree in enumerate(ptb.parsed_sents()): # [1360+1334:]
        # print(i)
        vocab += Counter(t.label() for t in tree.subtrees() if t.height() > 2)
        tree.collapse_unary()
        if len(tree) == 1:
            print('hahahahaha y', i)
            tree[0].set_label(tree.label() + '+' + tree[0].label())
            tree = tree[0]
            #tree.draw()
        for ind, leaf in reversed(list(enumerate(tree.leaves()))):
            leaf_path = tree.leaf_treeposition(ind)
            grandpath = leaf_path[:-2] # leaf must be unary
            if leaf.startswith("*") and tree[leaf_path[:-1]].label() != '-NONE-':
                print(leaf)
                return False
            if grandpath and len(tree[grandpath[:-1]]) == 1: # got no other child
                #tree[grandpath[:-1]].draw()
                #tree.draw()
                return False
            if len(grandpath) > 1:
                unary_trans[tree[grandpath].label()][tree[grandpath][0].label()] += 1
                # if t.label().startswith('ADVP-DIR'):
                #     tree.draw()
    if pickle_to:
        from pprint import pprint
        ret_dict = {}
        ret_dict['unary_cnt'] = unary_trans
        ret_dict['vocab_size'] = vocab
        pprint(ret_dict)
        from pickle import dump
        with open(pickle_to, 'wb') as fw:
            dump(ret_dict, fw)
    return True

def view_pickle(pickle_to):
    from pickle import load
    with open(pickle_to, 'rb') as fr:
        res = load(fr)
    d = defaultdict(int)
    for w, c in res['vocab_size'].items():
        if '=' in w:
            w = w.split('=')[0]
        if '-' in w:
            w = w.split('-')
            if w[-1].isdigit():
                w = w[:-1]
            d['-'.join(w)] += c
        else:
            d[w] += c
    print(d)
    print(len(d))