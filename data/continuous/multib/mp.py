from data.mp import DM
from data.continuous.multib import get_tree_from_signals

class MultibDM(DM):
    @staticmethod
    def tree_gen_fn(i2b, i2t, i2l, tag_layer, batch_segment, token, tag, label, chunk, segment):
        for args in zip(token, tag, label, chunk, segment):
            tree, _ = tensor_to_tree(i2b, i2t, i2l, tag_layer, batch_segment, *args, fallback_label = 'ROOT')
            tree.un_chomsky_normal_form()
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        start = seg_id * seg_size
        if start < batch_size:
            return args[:2] + tuple(x[start: (seg_id + 1) * seg_size] for x in args[2:])

def tensor_to_tree(i2b, i2t, i2l, tag_layer, batch_segment, bottom, tag, label, chunk, segment, weight = None, vote = None, **kwargs):
    layers_of_labels = []
    layers_of_chunks = []
    layers_of_weight = None if weight is None else []
    layers_of_vote   = None if vote   is None else []

    total_chunk = len(chunk)
    label_start = chunk_start = vote_start = weight_start = 0
    if tag_layer:
        wd = [i2b[c] for c in bottom[:segment[0]]]
        tg = [i2t[i] for i in    tag[:segment[tag_layer]]]
        nw = []
        for i in range(tag_layer):
            label_bnd = batch_segment[i]
            label_len = segment[i]
            char_chunk = chunk[chunk_start: chunk_start + label_len + 1].nonzero()[0]
            for start, end in zip(char_chunk, char_chunk[1:]):
                if weight is not None:
                    sg = ''.join(s + f'({w*100:.0f}%)' for s, w in zip(wd[start:end], weight[weight_start+start:, 0]))
                else:
                    sg = ''.join(wd[start:end])
                nw.append(sg)
            chunk_start  += label_bnd + 1
            weight_start += label_bnd
            vote_start   += label_bnd * (label_bnd + 1)
            wd = nw
        segment       = segment      [tag_layer:]
        batch_segment = batch_segment[tag_layer:]
    else:
        ln = segment[0]
        wd = [i2b[i] for i in bottom[:ln]]
        tg = [i2t[i] for i in    tag[:ln]]
        
    for l_size, l_len in zip(batch_segment, segment):
        layers_of_labels.append([i2l[i] for i in label[label_start: label_start + l_len]])
        if l_len == 1 or chunk_start < total_chunk and len(layers_of_chunks) > 1 and layers_of_chunks[-2].shape == layers_of_chunks[-1].shape:
            break
        layers_of_chunks.append(chunk[chunk_start: chunk_start + l_len + 1].nonzero()[0])
        if weight is not None:
            layers_of_weight.append(weight[weight_start: weight_start + l_len])
        weight_start += l_size
        label_start  += l_size
        chunk_start  += l_size + 1
        if vote is not None:
            vote_end = vote_start + (l_size + 1) * l_size
            vote_layer = vote[vote_start: vote_end]
            if vote_layer.size: # for erroneous outputs
                vote_layer = vote_layer.reshape(l_size + 1, l_size)[:l_len + 1, :l_len]
                layers_of_vote.append(vote_layer)
                vote_start = vote_end
    try:
        return get_tree_from_signals(wd, tg, layers_of_labels, layers_of_chunks, layers_of_weight, layers_of_vote, **kwargs)
    except:
        breakpoint()