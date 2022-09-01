from data import before_to_seq
from data.continuous.binary import get_tree_from_signals, after_to_tree, t_index, s_index

def head_to_tree(offset, length, tokens, tags, labels, rights, vocabs):
    tree, warn = get_tree_from_signals(*__before_to_tree(offset, length, tokens, tags, labels, rights, vocabs))
    assert len(warn) == 0
    return tree

# def to_tree(x): # slower than single
#     return head_to_tree(*x)


def data_to_tree(offset, length, tokens, tags, labels, rights, vocabs,
                 return_warnings = False,
                 on_warning      = None,
                 on_error        = None,
                 error_prefix    = ''):
    return after_to_tree(*__before_to_tree(offset, length, tokens, tags, labels, rights, vocabs),
                         return_warnings,
                         on_warning,
                         on_error,
                         error_prefix)

def triangle_to_layers(data, *size_offset_length_vocab):
    if size_offset_length_vocab:
        size, offset, length, vocab = size_offset_length_vocab
    else:
        length, offset = t_index(len(data))
        assert offset == 0, f'offset == {offset}'
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

def __before_to_tree(offset, length, tokens, tags, labels, rights, vocabs):
    size = len(tokens)
    i2w, i2t, label_vocab = before_to_seq(vocabs._nested)
    token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
    tag_layer   = tuple(i2t[t]  for t in tags [offset:offset+length]) if i2t else None
    label_layers = triangle_to_layers(labels, size, offset, length, label_vocab)
    right_layers = triangle_to_layers(rights, size, offset, length,        None)
    return token_layer, tag_layer, label_layers, right_layers

# def convert_batch(h, d, num_token, vocabs, fh, fd):

#     for i, l in enumerate(h.len):
#         if fh is not None:
#             tree = head_to_tree(h.token[i], h.tag[i], h.label[i], l, h.left[i], vocabs)
#             print(' '.join(str(tree).split()), file = fh)
#         tree, warnings = data_to_tree(h.token[i], d.tag[i], _label(i), l, _left(i), vocabs, return_warnings = True)
#         if fd is not None:
#             print(' '.join(str(tree).split()), file = fd)
#         yield i, l, warnings
# demands:
# 1. want to know whether there are warnings or errors and a safe results (e.g. individual visualization, calc whole scores)
# 2. suppress all the warnings and error (output to stderr), just get a safe result
# [4: 261], [5: 197], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683； relay
# [4: 598], [5: 998], [7: 12], [3: 19], [2: 3], [9: 3], [6: 11]/2415/56683： keep