from data.continuous.binary import get_tree_from_signals
from data import NIL

def trapezoid_to_layers(data, segments, seg_length, i2l = None, offset = 0):
    # assert data.shape[0] == sum(segments), f'slice shape not match ({data.shape[0]} vs. {sum(segments)})'
    start   = 0
    layers  = []
    for padded_length, length in zip(segments, seg_length):
        if length == 0: break
        layer_start = start + offset
        layer = data[layer_start : layer_start + length]
        if i2l is not None:
            if hasattr(layer, 'shape') and len(layer.shape) > 1:
                if i2l[0] == NIL:
                    layer = ['-'.join(i2l[y] for y in x if y > 0) for x in layer]
                else:
                    layer = ['-'.join(i2l[y] for y in x) for x in layer]
            else:
                layer = [i2l[x] for x in layer]
        layers.append(layer)
        start += padded_length
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

def data_to_tree(i2vs, offset, length, tokens, tags, labels, rights, seg_lengths, segments, **kwargs):
    if len(i2vs) == 3:
        i2w, i2t, i2l = i2vs
        tag_layer = tuple(i2t[t]  for t in tags [offset:offset+length])
    else:
        i2w, i2l = i2vs
        tag_layer = None
    token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
    label_layers = trapezoid_to_layers(labels, segments, seg_lengths,  i2l)
    right_layers = trapezoid_to_layers(rights, segments, seg_lengths, None)
    return get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers, **kwargs)