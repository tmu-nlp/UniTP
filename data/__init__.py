SUP = '#'
SUB = '_'

from os import listdir
types = tuple(f[:-3] for f in listdir('data') if f.endswith('_types.py'))

def brackets2RB(w):
    return {'(': '-LRB-', ')': '-RRB-'}.get(w, w)

def RB2brackets(w):
    return {'-LRB-': '(', '-RRB-': ')'}.get(w, w)

def XRB2brackets(word):
    if '\\' in word: # single \ in nltk.cp.tb
        return word.replace('\\', '')
    return RB2brackets(word)

def remove_eq(label, additional = None):
    pos = label.rfind('=')
    if pos > 0:
        label = label[:pos]
    elif additional:
        label = label[:label.rfind(additional)]
    return label

import numpy as np
from utils.types import NIL
def before_to_seq(vocabs):
    if 'tag' in vocabs: # label_mode
        i2t = vocabs['tag']
        label_vocab = vocabs['label'].__getitem__
    else:
        i2t = None
        if 'label' in vocabs:
            i2l = vocabs['label']
            label_vocab = lambda x: i2l[x]
        elif 'polar' in vocabs:
            i2p = vocabs['polar']
            def label_vocab(x):
                if isinstance(x, np.ndarray):
                    if x[0] < 0:
                        return NIL
                    return ''.join(i2p[xi] for xi in x)
                return NIL if x < 0 else i2p[x]
        else:
            label_vocab = lambda x: f'{x * 100:.2f}%' if x > 0 else NIL
    return vocabs['token'], i2t, label_vocab