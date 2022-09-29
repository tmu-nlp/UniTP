
from data.mp import DM, BaseVis
from data.cross import explain_error
from data.cross.evalb_lcfrs import export_string
from data.cross.binary import disco_tree as binary_tree
from data.cross.multib import disco_tree as multib_tree
from data.cross.evalb_lcfrs import DiscoEvalb, ExportWriter
from utils.shell_io import discodop_eval, byte_style

class BxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, batch_segment, *data_gen):
        for args in zip(*data_gen):
            bt, td, _ = binary_tree(*binary_layers(i2w, i2t, i2l, batch_segment, *args), 'VROOT')
            yield export_string(bid_offset, bt, td)
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

def b_batch_trees(bid_offset, data_gen, batch_segment, i2vs, fallback_label = None):
    for sid, args in enumerate(data_gen):
        sid += bid_offset
        largs = binary_layers(*i2vs, batch_segment, *args)
        # btm, tpd, err = 
        # children = {k for td in tpd.values() for k in td.children}
        # if any(b not in children for b,_,_ in btm):
        #     breakpoint()
        #     disco_tree(words, tags, layers_of_label, layers_of_xtype, layers_of_joint, fallback_label)
        try:
            yield binary_tree(*largs, fallback_label)
        except:
            breakpoint()
            binary_tree(*largs, fallback_label)
            binary_tree(*largs, fallback_label)

class MxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, batch_segment, *data_gen):
        for seg_length, word, tag, label, space in zip(*data_gen):
            layers_of_label = []
            layers_of_space = []
            label_start = 0
            for l_size, l_len in zip(batch_segment, seg_length):
                label_end = label_start + l_len
                label_layer = label[label_start: label_end]
                layers_of_label.append(tuple(i2l[i] for i in label_layer))
                if l_len == 1:
                    break
                layers_of_space.append(space[label_start: label_end])
                label_start += l_size
            ln = seg_length[0]
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


def m_batch_trees(b_word, b_tag, b_label, b_space, b_segment, b_seg_length, i2vs, fb_label = None, b_weight = None):
    add_weight = b_weight is not None
    for sid, (word, tag, label, space, seg_length) in enumerate(zip(b_word, b_tag, b_label, b_space, b_seg_length)):
        layers_of_label = []
        layers_of_space = []
        layers_of_weight = [] if add_weight else None
        label_start = 0
        for l_size, l_len in zip(b_segment, seg_length):
            label_end = label_start + l_len
            label_layer = label[label_start: label_end]
            layers_of_label.append(tuple(i2vs.label[i] for i in label_layer))
            if l_len == 1:
                break
            layers_of_space.append(space[label_start: label_end])
            if add_weight:
                layers_of_weight.append(b_weight[sid, label_start: label_end])
            label_start += l_size
        ln = seg_length[0]
        wd = [i2vs.token[i] for i in word[:ln]]
        tg = [i2vs.tag  [i] for i in  tag[:ln]]
        yield multib_tree(wd, tg, layers_of_label, layers_of_space, fb_label, layers_of_weight)


class DVA(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm):
        super().__init__(epoch, work_dir, i2vs)
        self._evalb = DiscoEvalb()
        self._logger = logger
        self._evalb_lcfrs_kwargs = evalb_lcfrs_kwargs
        self._discodop_prm = discodop_prm
        self._head_trees = head_trees
        self._xh_writer = ExportWriter() if not head_trees and discodop_prm else None
        self._xd_writer = ExportWriter() if discodop_prm else None
        self._v_errors = {}
        self._data_batch_cnt = 0

    @property
    def pending_head(self):
        return self._xh_writer is None

    def save_head_trees(self, *head_trees):
        self._head_trees.append(head_trees)

    def get_head_trees(self):
        head_trees = self._head_trees[self._data_batch_cnt]
        self._data_batch_cnt += 1
        return head_trees

    def _after(self):
        total_sents, num_errors = self._evalb.total_missing
        if num_errors:
            self._logger(f'  {num_errors} system errors from evalb (this should not appear in log)')
        
        if num_errors := len(self._v_errors):
            fname = f'data.{self.epoch}.errors'
            self._logger(f'  {num_errors} system errors, check {fname} for details.')
            with open(self.join(fname), 'w') as fw:
                for sid, error_args in self._v_errors.items():
                    fw.write(explain_error(*error_args) + '\n')

        if self._xh_writer:
            self._xh_writer.dump(self.join('head.export'))
        
        with open(self.join(f'eval.{self.epoch}.rpt'), 'w') as fw:
            fw.write(str(self._evalb))

            if self._xd_writer:
                fhead = self.join('head.export')
                fdata = self.join(f'data.{self.epoch}.export')
                self._xd_writer.dump(fdata)
                scores = discodop_eval(fhead, fdata, self._discodop_prm, fw)
                tp, tr, tf, dp, dr, df = (scores[k] for k in ('TP', 'TR', 'TF', 'DP', 'DR', 'DF'))
                scores['N'] = total_sents
            else:
                tp, tr, tf, dp, dr, df = self._evalb.summary()
                scores = dict(TP = tp, TR = tr, TF = tf, DP = dp, DR = dr, DF = df, N = total_sents)

        desc_for_screen = f'Evalb({tp:.2f}/{tr:.2f}/' + byte_style(f'{tf:.2f}', underlined = True)
        desc_for_screen += f'|{dp:.2f}/{dr:.2f}/' + byte_style(f'{df:.2f}', underlined = True) + ')'
        desc_for_logger = f'N: {total_sents} Evalb({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
        return scores, desc_for_screen, desc_for_logger, self._head_trees


class DVP(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, evalb_lcfrs_kwargs, discodop_prm, dm, corp_key):
        super().__init__(epoch, work_dir, i2vs)
        self._pending_heads = False
        assert discodop_prm
        self._v_errors = {}
        self._args = dm, discodop_prm, evalb_lcfrs_kwargs, corp_key
        self._bid_offset = 1

    def _before(self):
        self._args[0].timeit()

    def _after(self):
        fhead = self.join('head.export')
        fdata = self.join(f'data.{self.epoch}.export')
        dm, discodop_prm = self._args[:2]
        
        tree_text = dm.batched()
        if tree_text: # 'None' means 'text concat' without a memory travel
            with open(fdata, 'w') as fw:
                fw.write(tree_text)

        with open(self.join(f'eval.{self.epoch}.rpt'), 'w') as fw:
            scores = discodop_eval(fhead, fdata, discodop_prm, fw)

        scores['N'] = self._bid_offset
        tp, tr, tf, dp, dr, df = (scores[k] for k in ('TP', 'TR', 'TF', 'DP', 'DR', 'DF'))
        desc_for_screen = f'Evalb({tp:.2f}/{tr:.2f}/' + byte_style(f'{tf:.2f}', underlined = True)
        desc_for_screen += f'|{dp:.2f}/{dr:.2f}/' + byte_style(f'{df:.2f}', underlined = True) + ')'
        desc_for_logger = f'N: {self._bid_offset} Evalb({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
        return scores, desc_for_screen, desc_for_logger

from copy import deepcopy
from data.cross import bracketing, Counter, new_word_label, filter_words
def inner_score(bt, td, prm_args, export_writer = None):
    td = deepcopy(td)
    if export_writer:
        export_writer.add(bt, td)
    bt, td = new_word_label(bt, td, word_fn = prm_args.word_fn, label_fn = prm_args.label_fn)
    filter_words(bt, td, prm_args.DELETE_WORD)
    brac_cnt, brac_mul = bracketing(bt, td, excluded_labels = prm_args.DELETE_LABEL) if td else Counter()
    return brac_cnt, brac_mul, set(bt)