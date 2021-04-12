import torch
from torch import nn
from utils.operator import Operator
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, false_type
from models.utils import fraction
from models.loss import cross_entropy, binary_cross_entropy, hinge_loss
from experiments.helper import WarmOptimHelper
from utils.shell_io import byte_style

train_type = dict(loss_weight = dict(pos = BaseType(0.5, validator = frac_open_0),
                                     bio = BaseType(0.5, validator = frac_open_0),
                                     ner = BaseType(0.5, validator = frac_open_0),
                                     fence = BaseType(0.5, validator = frac_open_0)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  fence_hinge_loss = true_type)

class NerOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, train_config):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._sigmoid = nn.Sigmoid()
        self._initial_run = True, True
        self._train_config = train_config
        t2is = []
        for tok in i2vs.token:
            new_tok = ''
            for t in tok:
                if t == '(':
                    t = '<'
                elif t == ')':
                    t = '>'
                new_tok += t
            t2is.append(new_tok)
        self._t2is = tuple(t2is)

    def _build_optimizer(self, start_epoch):
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
        self.recorder.init_tensorboard()
        optim = hp.optimizer
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        self.recorder.tensorboard(self.global_step, 'Batch/%s', Epoch = epoch,
                                  Learning_Rate = self._schedule_lr(epoch, wander_ratio))

    def _step(self, mode, ds_name, batch, batch_id = None):

        if mode == M_TRAIN and 'fence' in batch:
            batch['supervised_fence'] = batch['fence']
            #(batch['offset'], batch['length'])

        batch_time = time()
        (batch_size, batch_len, existence, pos_logits, bio_logits, ner_logits, fence_logits, fences,
         weights) = self._model(**batch)
        batch_time = time() - batch_time

        if mode == M_TRAIN:
            losses = {}
            accuracies = {}
            if bio_logits is not None:
                losses['BIO'] = bio_loss = cross_entropy(bio_logits, batch['bio'])
                total_loss = bio_loss * self._train_config.loss_weight.bio
                accuracies['BIO'] = fraction(batch['bio'] == bio_logits.argmax(dim = 2), existence)
            else:
                losses['NER'] = ner_loss = cross_entropy(ner_logits, batch['ner'])
                total_loss = ner_loss * self._train_config.loss_weight.ner
                if self._train_config.fence_hinge_loss:
                    fence_loss = hinge_loss(fence_logits, batch['fence'], None)
                else:
                    fence_logits = self._sigmoid(fence_logits)
                    fence_loss = binary_cross_entropy(fence_logits, batch['fence'], None)
                losses['Fence'] = fence_loss
                total_loss = total_loss + fence_loss * self._train_config.loss_weight.fence
                accuracies['NER']   = fraction(batch['ner']   == ner_logits.argmax(dim = 2))
                accuracies['Fence'] = fraction(batch['fence'] == fences)
            if pos_logits is not None:
                losses['POS'] = pos_loss = cross_entropy(pos_logits, batch['pos'])
                accuracies['POS'] = fraction(batch['pos'] == pos_logits.argmax(dim = 2), existence)

            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s', **accuracies)
            self.recorder.tensorboard(self.global_step, 'Loss/%s', Total  = total_loss, **losses)
            self.recorder.tensorboard(self.global_step, 'Batch/%s', Length = batch_len, SamplePerSec = batch_len / batch_time)
        else:
            vis, _, _, flush_heads, save_tensors = self._vis_mode
            b_head = [batch_id, batch['length'].cpu().numpy(), batch['token'].cpu().numpy()]
            if flush_heads:
                for field in 'pos bio ner fence'.split():
                    if field == 'pos' and pos_logits is None:
                        b_head.append(None)
                    elif field in batch:
                        value = batch[field]
                        if field != 'fence':
                            value = value.type(torch.uint8)
                        b_head.append(value.cpu().numpy())
                    else:
                        b_head.append(None)
            b_head = tuple(b_head)
            pos = None if pos_logits is None else pos_logits.argmax(dim = 2).type(torch.uint8).cpu().numpy()
            bio = None if bio_logits is None else bio_logits.argmax(dim = 2).type(torch.uint8).cpu().numpy()
            # import pdb; pdb.set_trace()
            if ner_logits is None:
                ner = fence = weight = None
            else:
                ner = ner_logits.argmax(dim = 2).type(torch.uint8).cpu().numpy()
                fence = fences.cpu().numpy()
                weight = weights.mean(dim = 2).cpu().numpy() if save_tensors else None
            b_data = (pos, bio, ner, fence, weight)
            vis.process(b_head + b_data)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_init, test_init = self._initial_run
        if use_test_set:
            if final_test:
                folder = ds_name + '_test'
                save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(float(epoch)))
            flush_heads = test_init
            self._initial_run = devel_init, False
        else:
            folder = ds_name + '_devel'
            save_tensors = is_bin_times(int(float(epoch)))
            flush_heads = devel_init
            self._initial_run = False, test_init

        vis = NerVis(epoch,
                     self.recorder.create_join(folder),
                     self.recorder.evalb,
                     self.i2vs,
                     self._t2is,
                     self.recorder.log,
                     flush_heads)
        vis = VisRunner(vis, async_ = False) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, flush_heads, save_tensors

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, flush_heads, save_tensors = self._vis_mode
        scores, desc, logg = vis.after()
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        speed_outer = float(f'{count / seconds:.1f}')
        speed_inner = float(f'{count / vis.proc_time:.1f}') # unfolded with multiprocessing
        logg += f' @{speed_outer} ◇ {speed_inner} sps. (sym:nn {rate:.2f}; {seconds:.3f} sec.)'
        scores['speed'] = speed_outer
        if not final_test:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores.get('F1', 0), SamplePerSec = speed_outer)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        if scores['TA'] == 100: scores.pop('TA')
        scores['key'] = scores.get('F1', 0)
        return scores

    def optuna_model(self):
        pass

import numpy as np
from data.ner_types import bio_to_tree, ner_to_tree
def batch_trees(b_length, b_word, b_pos, b_bio, b_ner, b_fence, i2vs, b_weight = None, t2vs = None, **kw_args):
    has_pos = b_pos is not None
    has_bio = b_bio is not None
    show_internal = t2vs is None
    if show_internal:
        t2vs = i2vs.token
    for sid, (ln, word) in enumerate(zip(b_length, b_word)):
        wd = [t2vs    [i] for i in word [     :ln]]
        ps = [i2vs.pos[i] for i in b_pos[sid, :ln]] if has_pos else None
        
        if has_bio:
            bi = [i2vs.bio[i] for i in b_bio[sid, :ln]]
            yield bio_to_tree(wd, bi, ps, show_internal, **kw_args)
        else:
            nr = [i2vs.ner[i] for i in b_ner[sid, :b_fence[sid, 1:].sum()]]
            fence, = np.where(b_fence[sid]) #.nonzero() in another format
            weights = b_weight[sid, :ln] if show_internal and b_weight is not None else None
            tree = ner_to_tree(wd, nr, fence, ps, show_internal, weights, **kw_args)
            if len(tree) == 0:
                import pdb; pdb.set_trace()
            yield tree

from utils.vis import BaseVis, VisRunner
from data.multib import draw_str_lines
from utils.file_io import join, isfile, remove
from utils.shell_io import parseval, rpt_summary
from nltk.tree import Tree
class NerVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, t2is, logger, flush_heads = False):
        super().__init__(epoch)
        self._evalb_i2vs_logger = evalb, i2vs, t2is, logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._art_lines = []
        self._art = join(work_dir, f'data.{epoch}.art')
        self._err = join(work_dir, f'data.{epoch}.rpt')
        self._fnames = htree, dtree
        self._head_tree = flush_heads
        self._data_tree = None

    def _before(self):
        htree, dtree = self._fnames
        if self._head_tree:
            self._head_tree = open(htree, 'w')
        self._data_tree = open(dtree, 'w')

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _process(self, batch):
        evalb, i2vs, t2is, logger = self._evalb_i2vs_logger
        if self._head_tree:
            (batch_id, h_length, h_token, h_pos, h_bio, h_ner, h_fence,
             d_pos, d_bio, d_ner, d_fence, d_weight) = batch
            for tree in batch_trees(h_length, h_token, h_pos, h_bio, h_ner, h_fence, i2vs, t2vs = t2is):
                print(' '.join(str(tree).split()), file = self._head_tree)
        else:
            (batch_id, h_length, h_token,
             d_pos, d_bio, d_ner, d_fence, d_weight) = batch
        for tree in batch_trees(h_length, h_token, d_pos, d_bio, d_ner, d_fence, i2vs, t2vs = t2is):
            print(' '.join(str(tree).split()), file = self._data_tree)
        for tree in batch_trees(h_length, h_token, d_pos, d_bio, d_ner, d_fence, i2vs, d_weight, root_label = 'Prediction'):
            self._art_lines.append('\n'.join(draw_str_lines(tree)))

    def _after(self): # TODO TODO TODO length or num_ners
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        evalb, _, _, logger = self._evalb_i2vs_logger
        proc = parseval(evalb, *self._fnames)
        report = proc.stdout.decode()
        s_rows, scores = rpt_summary(report, True, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        if num_errors:
            logger(f'  {num_errors} errors from evalb')
            for e, error in enumerate(errors[:10]):
                logger(f'    {e}. ' + error)
            if num_errors > 10:
                logger('     ....')
            logger(f'  Go check {self._err} for details.')
        with open(self._err, 'w') as fw:
            fw.write(report)

        self._head_tree = self._data_tree = None
        with open(self._fnames[0]) as fr, open(self._art, 'w') as fw:
            for gold_tree, pred_tree, s_row in zip(fr, self._art_lines, s_rows):
                lines = f'Sent #{s_row[0]}:'
                if s_row[3] == s_row[4] == 0 or s_row[3] == s_row[4] == 100:
                    lines += ' EXACT MATCH\n' + pred_tree + '\n\n\n'
                else:
                    gold_tree = Tree.fromstring(gold_tree)
                    gold_tree.set_label('Gold')
                    lines += '\n' + '\n'.join(draw_str_lines(gold_tree))
                    lines += '\n\n' + pred_tree + '\n\n\n'
                fw.write(lines)
                    
        #  0     1      2    3     4        5      6   7       8    9      10     11
        #  ID  Len.  Stat. Recal  Prec.  Bracket gold test Bracket Words  Tags Accracy
        # ============================================================================
        #    1    2    0  100.00  50.00     1      1    2      0      2     1    50.00


        # if self.length_bins is not None and self._scores_of_bins:
        #     with open(self._ctvis.join(f'{self.epoch}.scores'), 'w') as fw:
        #         fw.write('wbin,num,lp,lr,f1,ta\n')
        #         for wbin in self.length_bins:
        #             fhead = self._ctvis.join(f'head.bin_{wbin}.tree')
        #             fdata = self._ctvis.join(f'data.bin_{wbin}.tree')
        #             proc = parseval(self._evalb, fhead, fdata)
        #             smy = rpt_summary(proc.stdout.decode(), False, True)
        #             fw.write(f"{wbin},{smy['N']},{smy['LP']},{smy['LR']},{smy['F1']},{smy['TA']}\n")
        #             remove(fhead)
        #             remove(fdata)

        desc = f'Evalb({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger