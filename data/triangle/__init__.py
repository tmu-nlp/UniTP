import numpy as np
from data.delta import NIL, t_index, s_index
from data.delta import get_tree_from_triangle, explain_warnings, explain_one_error
from nltk.tree import Tree
def head_to_tree(offset, length, words, tags, labels, rights, vocabs):
    tree, warn = get_tree_from_triangle(*__before_to_tree(offset, length, words, tags, labels, rights, vocabs))
    assert len(warn) == 0
    return tree

def data_to_tree(offset, length, words, tags, labels, rights, vocabs,
                 return_warnings = False,
                 on_warning      = None,
                 on_error        = None,
                 error_prefix    = ''):
    return after_to_tree(*__before_to_tree(offset, length, words, tags, labels, rights, vocabs),
                         return_warnings,
                         on_warning,
                         on_error,
                         error_prefix)

def triangle_to_layers(data, *size_offset_length_vocab):
    if size_offset_length_vocab:
        size, offset, length, vocab = size_offset_length_vocab
    else:
        length, offset = t_index(len(data))
        assert offset == 0
        size = length
        vocab = None

    pad_len = size - length
    layers = []
    for level in range(size):
        seq_len = level - pad_len
        if seq_len < 0:
            continue
        start = s_index(level) + offset
        end   = start + seq_len + 1
        layer = data[start:end]
        if vocab:
            layer = tuple(vocab(x) for x in layer)
        layers.append(layer)
    layers.reverse()
    return layers

def before_to_seq(offset, length, words, tags, labels, vocabs):
    word_layer      = tuple(vocabs.word[w] for w in words[offset:offset+length])
    if tags is not None: # label_mode
        tag_layer   = tuple(vocabs.tag[t]  for t in tags [offset:offset+length])
        label_vocab = vocabs.label.__getitem__
    else:
        tag_layer = None
        label_vocab = lambda x: NIL if x < 0 else vocabs.label[x]
    return word_layer, tag_layer, label_vocab

def after_to_tree(word_layer, tag_layer, label_layers, right_layers,
                    return_warnings = False,
                    on_warning      = None,
                    on_error        = None,
                    error_prefix    = ''):
    try:
        tree, warnings = get_tree_from_triangle(word_layer, tag_layer, label_layers, right_layers)
    except ValueError as e:
        error, last_layer, warnings = e.args
        if callable(on_error):
            on_error(error_prefix, explain_one_error(error))
        tree = Tree('S', [x for x in last_layer if x]) # Trust the model: TODO report failure rate
        warnings.append(error)
    if warnings and callable(on_warning) and tag_layer is not None:
        on_warning(explain_warnings(warnings, label_layers, tag_layer))
    if return_warnings: # [:, 2] > 8 is error
        warnings = np.asarray(warnings, dtype = np.int8)
        warnings.shape = (-1, 3)
        return tree, warnings
    return tree

def __before_to_tree(offset, length, words, tags, labels, rights, vocabs):
    size = len(words)
    word_layer, tag_layer, label_vocab = before_to_seq(offset, length, words, tags, labels, vocabs)
    label_layers = triangle_to_layers(labels, size, offset, length, label_vocab)
    right_layers = triangle_to_layers(rights, size, offset, length,        None)
    return word_layer, tag_layer, label_layers, right_layers

# def convert_batch(h, d, num_word, vocabs, fh, fd):

#     for i, l in enumerate(h.len):
#         if fh is not None:
#             tree = head_to_tree(h.word[i], h.tag[i], h.label[i], l, h.left[i], vocabs)
#             print(' '.join(str(tree).split()), file = fh)
#         tree, warnings = data_to_tree(h.word[i], d.tag[i], _label(i), l, _left(i), vocabs, return_warnings = True)
#         if fd is not None:
#             print(' '.join(str(tree).split()), file = fd)
#         yield i, l, warnings
# demands:
# 1. want to know whether there are warnings or errors and a safe results (e.g. individual visualization, calc whole scores)
# 2. suppress all the warnings and error (output to stderr), just get a safe result
# [4: 261], [5: 197], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683； relay
# [4: 598], [5: 998], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683： keep