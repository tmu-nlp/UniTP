from data.continuous.binary import get_tree_from_signals
from utils.math_ops import t_index
from data import NIL
    
def triangle_to_layers(data, *size_offset_length_vocab):
    if size_offset_length_vocab:
        padded_length, offset, length, i2l = size_offset_length_vocab
    else:
        padded_length, offset = t_index(len(data))
        assert offset == 0, f'offset == {offset}'
        length = padded_length
        i2l = None

    start = 0
    layers = []
    for inc in range(padded_length):
        if (layer_len := length - inc) == 0:
            break
        layer_start = start + offset
        layer = data[layer_start:layer_start + layer_len]
        if i2l is not None:
            if hasattr(layer, 'shape') and len(layer.shape) > 1:
                if i2l[0] == NIL:
                    layer = ['-'.join(i2l[y] for y in x if y > 0) for x in layer]
                else:
                    layer = ['-'.join(i2l[y] for y in x) for x in layer]
            else:
                layer = [i2l[x] for x in layer]
        layers.append(layer)
        start += padded_length - inc
    return layers

def data_to_tree(i2vs, offset, length, tokens, tags, labels, rights, **kwargs):
    size = len(tokens)
    if len(i2vs) == 3:
        i2w, i2t, i2l = i2vs
        tag_layer = tuple(i2t[t]  for t in tags [offset:offset+length])
    else:
        tag_layer = None
        i2w, i2l = i2vs
    token_layer  = tuple(i2w[w] for w in tokens[offset:offset+length])
    label_layers = triangle_to_layers(labels, size, offset, length,  i2l)
    right_layers = triangle_to_layers(rights, size, offset, length, None)
    return get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers, **kwargs)

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