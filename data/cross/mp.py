
from data.mp import DM
from data.cross.evalb_lcfrs import export_string, export_failed_string
from data.cross.binary import disco_tree as binary_tree
from data.cross.multib import disco_tree as multib_tree

class BxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, batch_segment, *data_gen):
        for segment, token, tag, label, xtype, joint in zip(*data_gen):
            bt, td, _ = binary_tree(*binary_layers(i2w, i2t, i2l, batch_segment, segment, token, tag, label, xtype, joint), 'VROOT')
            try:
                yield export_string(bid_offset, bt, td)
            except:
                yield export_failed_string(bid_offset, bt)
                from datetime import datetime
                from pprint import pprint
                with open('export.error.dbg.txt', 'a+') as fw:
                    fw.write(f'# {datetime.now()}: BxDM\n')
                    pprint(bt, fw)
                    pprint(td, fw)
            bid_offset += 1

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        bid_offset, segment = args[:2]
        start = seg_id * seg_size
        if start < batch_size:
            return (bid_offset + start, segment) + tuple(x[start: (seg_id + 1) * seg_size] for x in args[2:])

def binary_layers(i2w, i2t, i2l, batch_segment, segment, token, tag, label, xtype, joint):
    layers_of_label = []
    layers_of_xtype = []
    layers_of_joint = []
    jnt_start = rgt_start = 0
    for s_size, s_len in zip(batch_segment, segment):
        label_layer = tuple(i2l[i] for i in label[rgt_start + 1: rgt_start + s_len + 1])
        layers_of_label.append(label_layer)
        layers_of_joint.append({i for i, j in enumerate(joint[jnt_start: jnt_start + s_len]) if i and j})
        layers_of_xtype.append(xtype[rgt_start + 1: rgt_start + s_len + 1])
        rgt_start += s_size
        jnt_start += s_size - 1
        if s_len == 1:
            break
    bottom_end = segment[0] + 1
    tags  = tuple(i2t[i] for i in   tag[1:bottom_end])
    words = tuple(i2w[i] for i in token[1:bottom_end])
    return words, tags, layers_of_label, layers_of_xtype, layers_of_joint

class MxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, batch_segment, *data_gen):
        for segment, word, tag, label, space in zip(*data_gen):
            layers_of_label = []
            layers_of_space = []
            label_start = 0
            for l_size, l_len in zip(batch_segment, segment):
                label_end = label_start + l_len
                label_layer = label[label_start: label_end]
                layers_of_label.append(tuple(i2l[i] for i in label_layer))
                if l_len == 1:
                    break
                layers_of_space.append(space[label_start: label_end])
                label_start += l_size
            ln = segment[0]
            wd = [i2w[i] for i in word[:ln]]
            tg = [i2t[i] for i in  tag[:ln]]
            bt, td, _ = multib_tree(wd, tg, layers_of_label, layers_of_space, 'VROOT')
            yield export_string(bid_offset, bt, td)
            bid_offset += 1

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, t_args):
        bid_offset, batch_segment = t_args[:2]
        start = seg_id * seg_size
        if start < batch_size:
            return (bid_offset + start, batch_segment) + tuple(x[start: (seg_id + 1) * seg_size] for x in t_args[2:])

def b_batch_trees(token, tag, label, xtype, joint, batch_segment, segment, i2vs, fb_label = None, perserve_sub = False):
    for largs in zip(segment, token, tag, label, xtype, joint):
        largs = binary_layers(*i2vs, batch_segment, *largs)
        # btm, tpd, err = 
        # children = {k for td in tpd.values() for k in td.children}
        # if any(b not in children for b,_,_ in btm):
        #     breakpoint()
        yield binary_tree(*largs, fb_label, perserve_sub)

def m_batch_trees(b_word, b_tag, b_label, b_space, batch_segment, segment, i2vs, fb_label = None, b_weight = None):
    add_weight = b_weight is not None
    for sid, (word, tag, label, space, segment) in enumerate(zip(b_word, b_tag, b_label, b_space, segment)):
        layers_of_label = []
        layers_of_space = []
        layers_of_weight = [] if add_weight else None
        label_start = 0
        for l_size, l_len in zip(batch_segment, segment):
            label_end = label_start + l_len
            label_layer = label[label_start: label_end]
            layers_of_label.append(tuple(i2vs.label[i] for i in label_layer))
            if l_len == 1:
                break
            layers_of_space.append(space[label_start: label_end])
            if add_weight:
                layers_of_weight.append(b_weight[sid, label_start: label_end])
            label_start += l_size
        ln = segment[0]
        wd = [i2vs.token[i] for i in word[:ln]]
        tg = [i2vs.tag  [i] for i in  tag[:ln]]
        yield multib_tree(wd, tg, layers_of_label, layers_of_space, fb_label, layers_of_weight)