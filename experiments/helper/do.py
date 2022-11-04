
from data.mp import BaseVis
from data.cross.evalb_lcfrs import read_param
from data.cross import explain_error, draw_str_lines
from data.cross.evalb_lcfrs import DiscoEvalb, ExportWriter
from utils.operator import Operator
from utils.shell_io import has_discodop, discodop_eval, byte_style
from utils.param_ops import get_sole_key
from experiments.helper import speed_logg, WarmOptimHelper, discontinuous_score_desc_logg, write_multilingual
from collections import namedtuple
HDIO = namedtuple('HDIO', 'bid_offset, evalb_lines, head_lines, trees_and_errors')

class DO(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, get_dm, train_config, evalb_lcfrs_prm):
        if has_discodop():
            prompt = 'Use discodop evalb (detected)'
            color = '2'
            self._discodop_prm = evalb_lcfrs_prm
        else:
            prompt = 'Use our dccp evalb, [discodop] is not installed.'
            prompt += '\n  => try \'pip install -r requirements.txt; make install\' for discodop>=0.65 (github version).'
            color = '3'
            if callable(get_dm): # TODO remove this
                prompt += '\n  [WARNING] \'multiprocessing_decode\' supports only discodop.'
                get_dm = None
            self._discodop_prm = None
        print(byte_style(prompt, color)); recorder.log(prompt)
        
        super().__init__(model, get_datasets, recorder, i2vs, get_dm)
        self._train_config = train_config
        self._tune_pre_trained = False
        self._evalb_lcfrs_kwargs = read_param(evalb_lcfrs_prm)
        self._init_mode_trees()

    def _init_mode_trees(self):
        if self.multi_corp:
            make = lambda v: {k: v() for k in self.i2vs}
            self._mode_trees = make(list), make(list)
        else:
            self._mode_trees = [], []

    def _build_optimizer(self, start_epoch):
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
        optim = hp.optimizer
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained.from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.tune_pre_trained.lr_factor if tune else 1
        learning_rate = self._schedule_lr(epoch, wander_ratio, lr_factor)
        self.recorder.tensorboard(self.global_step, 'Batch/%s', Learning_Rate = learning_rate, Epoch = epoch)

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial = self._vis_mode
        if serial:
            scores, desc, logg, heads = vis.after()
            devel_head, test_head = self._mode_trees
            if self.multi_corp:
                if use_test_set:
                    test_head [ds_name] = heads
                else:
                    devel_head[ds_name] = heads
            else:
                if use_test_set:
                    self._mode_trees = devel_head, heads
                else:
                    self._mode_trees = heads, test_head
        else:
            scores, desc, logg = vis.after()
        if self.multi_corp:
            desc = ds_name.upper() + desc
        else:
            desc = 'Evalb' + desc

        _desc, _logg, speed_outer, speed_dm = speed_logg(count, seconds, None if serial else self._dm)
        if not final_test:
            prefix = 'TestSet' if use_test_set else 'DevelSet'
            suffix = ds_name if self.multi_corp else None
            self.recorder.tensorboard(self.global_step, prefix + '/%s', suffix,
                                      F1 = scores.get('TF', 0), DF = scores.get('DF', 0),
                                      SamplePerSec = None if serial else speed_dm)
        scores['speed'] = speed_outer
        self._vis_mode = final_test
        return scores, desc + _desc, logg + _logg

    def combine_scores_and_decide_key(self, epoch, ds_scores):
        if self._vis_mode and self.multi_corp:
            write_multilingual(self._recorder.create_join('multilingual'), self.i2vs, self._model)
        if self.multi_corp:
            key = [s.get('TF', 0) for s in ds_scores.values()]
            ds_scores['key'] = sum(key) / len(key)
            return ds_scores
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores.get('TF', 0.0) #f_score(scores.get('TF', 0.0), scores.get('DF', 0.0))
        return scores

tag_root_fn = lambda i,t: t[i].label if i else f'{t[i].label} (Gold)'
from utils.file_io import remove, isdir, mkdir, listdir, join
class DVA(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, draw_trees):
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
        if draw_trees:
            from data.cross import draw_str_lines
            if isdir(draw_trees := self.join(f'tree.{epoch}.art')):
                for fname in listdir(draw_trees):
                    remove(join(draw_trees, fname))
            else:
                mkdir(draw_trees)
            draw_trees = draw_trees, draw_str_lines
        self._draw_trees = draw_trees

    @property
    def pending_head(self):
        return self._xh_writer is None

    def head_data_io(self, batch_id, trees, tree_gen, batch_args):
        if self._xh_writer:
            head_lines, head_trees_for_scores = [], []
            for bt, td in trees:
                head_trees_for_scores.append(
                    inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xh_writer))
                head_lines.append(
                    '\n'.join(draw_str_lines(bt, td, label_fn = tag_root_fn)))
            self.save_head_trees(head_trees_for_scores, head_lines)
        else: # TODO self.evalb make error here when testing!!!
            head_trees_for_scores, head_lines = self.get_head_trees()

        bid_offset, _ = self._evalb.total_missing
        self._evalb.add_batch_line(batch_id)
        evalb_lines, trees_and_errors = [], []
        for sid, (btm, tpd, error) in enumerate(tree_gen(*batch_args, self.i2vs, 'VROOT')):
            trees_and_errors.append((btm, tpd, error))
            try: # TODO error when multilingual
                pred = inner_score(btm, tpd, self._evalb_lcfrs_kwargs, self._xd_writer)
            except:
                from pprint import pprint
                pprint(btm)
                pprint(tpd)
                breakpoint()
            try:
                evalb_lines.append(self._evalb.add(*pred, *head_trees_for_scores[sid]))
            except:
                from pprint import pprint
                pprint(pred)
                pprint(head_trees_for_scores[sid])
                breakpoint()
            if error: self._v_errors[bid_offset + sid] = error
        return HDIO(bid_offset, evalb_lines, head_lines, trees_and_errors)

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

        desc_for_screen, desc_for_logger = discontinuous_score_desc_logg(tp, tr, tf, dp, dr, df)
        desc_for_logger = f'N: {total_sents} ' + desc_for_logger
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
        desc_for_screen, desc_for_logger = discontinuous_score_desc_logg(tp, tr, tf, dp, dr, df)
        desc_for_logger = f'N: {self._bid_offset} ' + desc_for_logger
        return scores, desc_for_screen, desc_for_logger

from copy import deepcopy
from data.cross import bracketing, Counter, new_word_label, filter_words
def inner_score(bt, td, prm_args, export_writer = None):
    if export_writer: export_writer.add(bt, td)
    bt, td = new_word_label(bt, deepcopy(td), word_fn = prm_args.word_fn, label_fn = prm_args.label_fn)
    filter_words(bt, td, prm_args.DELETE_WORD)
    brac_cnt, brac_mul = bracketing(bt, td, excluded_labels = prm_args.DELETE_LABEL) if td else Counter()
    return brac_cnt, brac_mul, set(bt)