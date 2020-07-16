from data.triangle import before_to_seq, after_to_tree
from data.delta import get_tree_from_triangle, explain_warnings, explain_one_error
def trapezoid_to_layers(data, segments, seg_length, vocab = None, offset = 0, big_endian = True):
    layers  = []
    l_end   = len(data)
    seg_len = list(zip(segments, seg_length))
    while seg_len:
        size, seq_len = seg_len.pop()
        l_start = l_end - size + offset
        layer = data[l_start:l_start + seq_len]
        if vocab:
            layer = tuple(vocab(x) for x in layer)
        layers.append(layer)
        if big_endian and seq_len == 1:
            break
        l_end -= size
    if not big_endian:
        layers.reverse()
    return layers

def inflate(layers):
    inflated = []
    expected_len = 1
    # import pdb; pdb.set_trace()
    while layers:
        inc = layers.pop()
        while len(inc) > expected_len:
            expected_len += 1
            inflated.append(None)
        expected_len += 1
        inflated.append(inc)
    inflated.reverse()
    return inflated

def head_to_tree(offset, length, tokens, tags, labels, rights, seg_lengths, segments, vocabs):
    args = __before_to_tree(offset, length, tokens, tags, labels, rights, segments, seg_lengths, vocabs)
    tree, warn = get_tree_from_triangle(*args)
    assert len(warn) == 0
    return tree

def data_to_tree(offset, length, tokens, tags, labels, rights, seg_lengths, segments, vocabs,
                 return_warnings = False,
                 on_warning      = None,
                 on_error        = None,
                 error_prefix    = '',
                 error_root      = 'S'):
    return after_to_tree(*__before_to_tree(offset, length, tokens, tags, labels, rights, segments, seg_lengths, vocabs),
                         return_warnings,
                         on_warning,
                         on_error,
                         error_prefix,
                         error_root)

def __before_to_tree(offset, length, tokens, tags, labels, rights, segments, seg_lengths, vocabs):
    token_layer, tag_layer, label_vocab = before_to_seq(offset, length, tokens, tags, vocabs)
    label_layers = trapezoid_to_layers(labels, segments, seg_lengths, label_vocab)
    right_layers = trapezoid_to_layers(rights, segments, seg_lengths,        None)
    return token_layer, tag_layer, label_layers, right_layers