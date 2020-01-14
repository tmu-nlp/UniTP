import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from utils.operator import Operator
from data.delta import get_rgt, get_dir, s_index
from data.penn_types import C_ABSTRACT
from time import time
from math import exp
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN
from models.utils import PCA, fraction
from models.loss import binary_cross_entropy, hinge_loss

class PennOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, evalb):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._evalb = evalb
        self._sigmoid = nn.Sigmoid()
        self._orient_hinge_loss = True
        self._mode_length_bins = None, None

    def _build_optimizer(self):
        self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.6
        self._writer = SummaryWriter(self.recorder.create_join('train'))
        self._last_wander_ratio = 0
        self._base_lr = 0.001
        # self._lr_discount_rate = 0.0001
        # for params in self._model.parameters():
        #     if len(params.shape) > 1:
        #         nn.init.xavier_uniform_(params)
        return optim.Adam(self._model.parameters(), betas = (0.9, 0.98), weight_decay = 5e-4)

    def _schedule(self, epoch, wander_ratio):
        wander_threshold = 0.15

        if wander_ratio < wander_threshold:
            learning_rate = self._base_lr * (1 - exp(- epoch))
        else:
            # lr_discount = self._base_lr * self._lr_discount_rate
            # if abs(self._last_wander_ratio - wander_ratio) > 1e-10: # change
            #     self._last_wander_ratio = wander_ratio
            #     if self._base_lr > lr_discount + 1e-10:
            #         self._base_lr -= lr_discount
            #     else:
            #         self._base_lr *= self._lr_discount_rate

            # if epoch > lr_half_life:
            #     base_lr *= 
            linear_dec = (1 - (wander_ratio - wander_threshold) / (1 - wander_threshold + 1e-20))
            learning_rate = self._base_lr * linear_dec

        # lr_half_life = 60
        # if epoch > lr_half_life:
        #     learning_rate *= exp(-(epoch - lr_half_life) / lr_half_life) # fine decline
        # learning_rate += 1e-20
        
        self._writer.add_scalar('Batch/Learning_Rate', learning_rate, self.global_step)
        self._writer.add_scalar('Batch/Epoch', epoch, self.global_step)
        for opg in self.optimizer.param_groups:
            opg['lr'] = learning_rate

    def _step(self, mode, ds_name, batch, flush = True, batch_id = None):
        # assert ds_name == C_ABSTRACT
        if mode == M_TRAIN and flush:
            self.optimizer.zero_grad()

        batch_time = time()
        gold_orients = get_rgt(batch['xtype'])
        if mode == M_TRAIN:
            batch['supervised_orient'] = gold_orients
            #(batch['offset'], batch['length'])
            
        (batch_size, batch_len, static, dynamic, layers_of_base, existences, orient_logits, tag_logits, label_logits,
         trapezoid_info) = self._model(batch['word'], **batch)
        batch_time = time() - batch_time
        
        orient_logits.squeeze_(dim = 2)
        existences   .squeeze_(dim = 2)
        if self._orient_hinge_loss:
            orients = orient_logits > 0
        else:
            orient_logits = self._sigmoid(orient_logits)
            orients = orient_logits > 0.5

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, -batch_len:]
            orient_weight = get_dir(batch['xtype'])
            tag_mis       = (tags    != batch['tag'])
            label_mis     = (labels  != batch['label'])
            orient_match  = (orients == gold_orients) & orient_weight
            tag_weight    = (   tag_mis | bottom_existence)
            label_weight  = ( label_mis | existences)

            if trapezoid_info is None:
                height_mask = s_index(batch_len - batch['length'])[:, None, None]
            else:
                height_mask = batch['mask_length'] # ?? negative effect ???

            tag_loss   = self._model.get_loss(tag_logits,   batch, None)
            label_loss = self._model.get_loss(label_logits, batch, height_mask)
            if self._orient_hinge_loss:
                orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
            else:
                orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

            total_loss = (tag_loss, label_loss, orient_loss)
            total_loss = zip(self._loss_weights_of_tag_label_orient, total_loss)
            total_loss = sum(w * loss for w, loss in total_loss)
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            if flush:
                self.optimizer.step()
            gs = self.global_step
            self._writer.add_scalar('Accuracy/Tag',     1 - fraction(tag_mis,    tag_weight),   gs)
            self._writer.add_scalar('Accuracy/Label',   1 - fraction(label_mis,  label_weight), gs)
            self._writer.add_scalar('Accuracy/Orient',  fraction(orient_match, orient_weight),  gs)
            self._writer.add_scalar('Loss/Tag',     tag_loss,    gs)
            self._writer.add_scalar('Loss/Label',   label_loss,  gs)
            self._writer.add_scalar('Loss/Orient',  orient_loss, gs)
            self._writer.add_scalar('Loss/Total',   total_loss,  gs)
            self._writer.add_scalar('Batch/SamplePerSec', batch_len / batch_time,  gs)
            self._writer.add_scalar('Batch/Length', batch_len,   gs)
            if 'segment' in batch:
                self._writer.add_scalar('Batch/Height', len(batch['segment']), gs)
        else:
            vis, _ = self._vis_mode
            mpc_word = mpc_label = None
            if vis.save_tensors:
                if hasattr(self._model._input_layer, 'pca'):
                    if dynamic is not None: # even dynamic might be None, being dynamic is necessary to train a good model
                        mpc_word = self._model._input_layer.pca(static, flush = flush)
                    mpc_label    = self._model._input_layer.pca(layers_of_base)
                else:
                    mpc_label = PCA(layers_of_base[:, -batch_len:].reshape(-1, layers_of_base.shape[2]))(layers_of_base)

                tag_scores,   tags   = self._model.get_decision_with_value(tag_logits)
                label_scores, labels = self._model.get_decision_with_value(label_logits)
                if self._orient_hinge_loss: # otherwise with sigmoid
                    orient_logits += 1
                    orient_logits /= 2
                    orient_logits[orient_logits < 0] = 0
                    orient_logits[orient_logits > 1] = 1
                b_mpcs = (None if mpc_word is None else mpc_word.type(torch.float16), mpc_label.type(torch.float16))
                b_scores = (tag_scores.type(torch.float16), label_scores.type(torch.float16), orient_logits.type(torch.float16))
            else:
                tags    = self._model.get_decision(tag_logits  )
                labels  = self._model.get_decision(label_logits)
                b_mpcs = (mpc_word, mpc_label)
                b_scores = (None, None, None)
            b_size = (batch_len,)
            b_head = tuple(batch[x].type(torch.uint8) if x in ('tag', 'label') else batch[x] for x in 'offset length word tag label'.split())
            b_head = b_head + (gold_orients,)
            b_logits = (tags.type(torch.uint8), labels.type(torch.uint8), orients)
            b_data = b_logits + b_mpcs + b_scores
            tensors = b_size + b_head + b_data
            tensors = tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensors)
            if trapezoid_info is not None:
                d_seg, d_seg_len = trapezoid_info
                trapezoid_info = batch['segment'], batch['seg_length'], d_seg, d_seg_len.cpu().numpy()
            vis.process(batch_id, tensors, trapezoid_info)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = True):
        devel_bins, test_bins = self._mode_length_bins
        if use_test_set:
            if final_test:
                folder = 'vis_test'
            else:
                folder = 'vis_test_with_devel'
            save_tensors = True
            length_bins = test_bins
            scores_of_bins = True
        else:
            folder = 'vis_devel'
            length_bins = devel_bins
            save_tensors = is_bin_times(int(float(epoch)) - 1)
            scores_of_bins = False

        self._model.eval()
        vis = PennVis(epoch,
                      self.recorder.create_join(folder),
                      self._evalb,
                      self.i2vs,
                      self.recorder.log,
                      save_tensors,
                      length_bins,
                      scores_of_bins)
        vis = VisRunner(vis, async_ = True) # wrapper
        vis.before()
        length_bins = vis.length_bins
        if length_bins is not None:
            if use_test_set:
                self._mode_length_bins = devel_bins, length_bins # change test
            else:
                self._mode_length_bins = length_bins, test_bins # change devel
        self._vis_mode = vis, use_test_set

    def _after_validation(self, speed):
        vis, use_test_set = self._vis_mode
        scores, desc, logg = vis.after()
        logg += ' @{speed:.2f}sps.'
        scores['speed'] = float(f'{speed:.1f}')
        scores[ 'key' ] = scores.get('F1', 0)
        if not use_test_set:
            self._writer.add_scalar('Evalb/F1', scores.get('F1', 0), self.global_step)
            self._writer.add_scalar('Evalb/SamplePerSec', speed,     self.global_step)
        self._vis_mode = None
        self._model.train()
        return scores, desc, logg

from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from visualization import set_vocab, set_head, set_data
class PennVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 save_tensors   = True,
                 length_bins    = None,
                 scores_of_bins = False):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._evalb = evalb
        self._i2vs = i2vs
        self._logger = logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{self.epoch}.tree')
        self._fnames = htree, dtree
        self._head_tree = None
        self._data_tree = None
        self._scores_of_bins = scores_of_bins
        self.register_property('save_tensors', save_tensors)
        self.register_property('length_bins',  length_bins)

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _before(self):
        htree, dtree = self._fnames
        if set_vocab(self._work_dir, self._i2vs._nested):
            # for fname in listdir(self._work_dir):
            #     if fname.startswith('head.'):
            #         fname = join(self._work_dir, fname)
            #         remove(fname)
            #     if fname.startswith('data.'):
            self._head_tree = open(htree, 'w')
            self.register_property('length_bins', set())
        self._data_tree = open(dtree, 'w')

    def _process(self, batch_id, batch, trapezoid_info):
        # process batch to instances, compress
        # if head is in batch, save it
        # else check head.emm.pkl
        # make data.emmb.tree & concatenate to data.emm.tree
        # make data.emm.rpt
        (size, h_offset, h_length, h_word, h_tag, h_label, h_right, 
         d_tag, d_label, d_right, mpc_word, mpc_label,
         tag_score, label_score, split_score) = batch
        d_trapezoid_info = None
        if trapezoid_info:
            segment, seg_length, d_segment, d_seg_length = trapezoid_info
            trapezoid_info = segment, seg_length
            d_trapezoid_info = d_segment, d_seg_length
            
        fpath = self._work_dir if self.save_tensors else None
        if self._head_tree:
            bins = set_head(fpath, batch_id,
                            size, h_offset, h_length, h_word, h_tag, h_label, h_right,
                            trapezoid_info,
                            self._i2vs, self._head_tree)
            if fpath:
                self.length_bins |= bins
        
        if self.length_bins is not None and self._scores_of_bins:
            bin_width = 10
        else:
            bin_width = None
            
        set_data(fpath, batch_id, size, self.epoch, 
                 h_offset, h_length, h_word, d_tag, d_label, d_right,
                 mpc_word, mpc_label,
                 tag_score, label_score, split_score,
                 d_trapezoid_info,
                 self._i2vs, self._data_tree, self._logger, self._evalb, bin_width)

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        proc = parseval(self._evalb, *self._fnames)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        fname = None
        if num_errors:
            self._logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    self._logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'

        self._head_tree = self._data_tree = None

        if self.length_bins is not None and self._scores_of_bins:
            fname = f'data.{self.epoch}.rpt'
            with open(join(self._work_dir, f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self.length_bins:
                    fhead = join(self._work_dir, f'head.bin_{wbin}.tree')
                    fdata = join(self._work_dir, f'data.bin_{wbin}.tree')
                    proc = parseval(self._evalb, fhead, fdata)
                    smy = rpt_summary(proc.stdout.decode(), False, True)
                    fw.write(f"{wbin},{smy['N']},{smy['LP']},{smy['LR']},{smy['F1']},{smy['TA']}\n")
                    remove(fhead)
                    remove(fdata)

        if fname:
            with open(join(self._work_dir, fname), 'w') as fw:
                fw.write(report)
            self._logger(f'  Go check {fname} for details.')

        desc = f'Evalb({scores["LP"]:.2f}/{scores["LR"]:.2f}/{scores["F1"]:.2f})'
        return scores, desc, f'N: {scores["N"]} {desc}'

# an example of Unmatched Length from evalb
# head
# (S (S (VP (VBG CLUBBING) (NP (DT A) (NN FAN)))) (VP (VBD was) (RB n't) (NP (NP (DT the) (NNP Baltimore) (NNP Orioles) (POS ')) (NN fault))) (. .))
# (S (NP (NP (JJ CLUBBING) (NNP A)) ('' FAN)) (VP (VBD was) (PP (RB n't) (NP     (DT the) (NNP Baltimore) (NNS Orioles) (POS ') (NN fault)))) (. .))
# data