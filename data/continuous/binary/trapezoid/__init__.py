from data.continuous.binary import get_tree_from_signals, after_to_tree
from data import before_to_seq

def trapezoid_to_layers(data, segments, seg_length, vocab = None, offset = 0, big_endian = True):
    # assert data.shape[0] == sum(segments), f'slice shape not match ({data.shape[0]} vs. {sum(segments)})'
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

def inflate(layers, reversed = True):
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
    if reversed:
        inflated.reverse()
    return inflated

def head_to_tree(offset, length, tokens, tags, labels, rights, seg_lengths, segments, vocabs):
    args = __before_to_tree(offset, length, tokens, tags, labels, rights, segments, seg_lengths, vocabs)
    tree, warn = get_tree_from_signals(*args)
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
    i2w, i2t, label_vocab = before_to_seq(vocabs._nested)
    token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
    tag_layer   = tuple(i2t[t]  for t in tags [offset:offset+length]) if i2t else None
    label_layers = trapezoid_to_layers(labels, segments, seg_lengths, label_vocab)
    right_layers = trapezoid_to_layers(rights, segments, seg_lengths,        None)
    return token_layer, tag_layer, label_layers, right_layers