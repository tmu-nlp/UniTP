SUP = '#'
SUB = '_'
SUBS = SUB + SUP
USUB = SUB + 'SUB'
NIL, PAD, UNK, BOS, EOS = '<nil>', '<pad>', '<unk>', '<bos>', '<eos>'


from os import listdir
types = tuple(f[:-3] for f in listdir('data') if f.endswith('_types.py'))

def brackets2RB(w):
    return {'(': '-LRB-', ')': '-RRB-'}.get(w, w)

def RB2brackets(w):
    return {'-LRB-': '(', '-RRB-': ')'}.get(w, w)

def no_backslash(word):
    return word.replace('\\', '')

def XRB2brackets(word):
    return RB2brackets(word)

def remove_eq(label, additional = None):
    pos = label.rfind('=')
    if pos > 0:
        label = label[:pos]
    elif additional:
        label = label[:label.rfind(additional)]
    return label

import numpy as np
def parsing_i2vs(vocabs):
    return vocabs.token, vocabs.tag, vocabs.label

def sentiment_i2vs(vocabs):
    return vocabs.token, vocabs.polar

# def 
#     if 'tag' in vocabs: # label_mode
#         i2t = vocabs['tag']
#         i2l = vocabs['label']
#     else:
#         i2t = None
#         if 'label' in vocabs:
#             i2l = vocabs['label']
#         elif 'polar' in vocabs:
#             i2p = vocabs['polar']
#             def label_vocab(x):
#                 if isinstance(x, np.ndarray):
#                     if x[0] < 0:
#                         return NIL
#                     return ''.join(i2p[xi] for xi in x)
#                 return NIL if x < 0 else i2p[x]
#         else:
#             label_vocab = lambda x: f'{x * 100:.2f}%' if x > 0 else NIL
#     return vocabs['token'], i2t, i2l