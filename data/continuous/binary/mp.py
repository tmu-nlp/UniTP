from data.mp import DM
from data.continuous.binary.trapezoid import trapezoid_to_layers
from data.continuous.binary.triangle import triangle_to_layers
from data.continuous.binary import get_tree_from_signals
class TriangularDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, offsets, lengths, token, tag, label, right):
        for offset, length, tokens, tags, labels, rights in zip(offsets, lengths, token, tag, label, right):
            size = len(tokens)
            token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
            tag_layer   = tuple(i2t[t] for t in tags  [offset:offset+length])
            label_layers = triangle_to_layers(labels, size, offset, length, i2l)
            right_layers = triangle_to_layers(rights, size, offset, length, None)
            tree, _ = get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers, 'VROOT')
            tree.un_chomsky_normal_form()
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        start = seg_id * seg_size
        if start < batch_size:
            return tuple(x[start: (seg_id + 1) * seg_size] for x in args)



class TrapezoidalDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, segments, offsets, lengths, token, tag, label, right, seg_length):
        for offset, length, tokens, tags, labels, rights, seg_length in zip(offsets, lengths, token, tag, label, right, seg_length):
            token_layer = tuple(i2w[w] for w in tokens[offset:offset+length])
            tag_layer   = tuple(i2t[t] for t in tags  [offset:offset+length])
            label_layers = trapezoid_to_layers(labels, segments, seg_length, i2l)
            right_layers = trapezoid_to_layers(rights, segments, seg_length, None)
            tree, _ = get_tree_from_signals(token_layer, tag_layer, label_layers, right_layers, 'VROOT')
            tree.un_chomsky_normal_form()
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        start = seg_id * seg_size
        if start < batch_size:
            return args[:1] + tuple(x[start: (seg_id + 1) * seg_size] for x in args[1:])