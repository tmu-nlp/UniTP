import torch
from torch import nn
from utils.operator import Operator
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06, frac_close
from models.utils import PCA, fraction, hinge_score, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper import warm_adam
from experiments.t_lstm_nccp.operator import PennOperator
from utils.shell_io import byte_style

train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     fence = BaseType(0.5, validator = frac_open_0)),
                  keep_low_attention_rate = BaseType(1.0, validator = frac_close),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained_from_nth_epoch = tune_epoch_type,
                  lr_factor_for_tuning = frac_06,
                  fence_hinge_loss = true_type)


def unpack_label_like(seq, segment):
    layers = []
    start = 0
    for size in segment:
        end = start + size
        layers.append(seq[:, start:end])
        start = end
    return layers

def unpack_fence(seq, segment, is_indices):
    layers = []
    start = 0
    for size in segment[is_indices:]:
        end = start + size + 1
        layers.append(seq[:, start:end])
        start = end
    return layers

def extend_fence_idx(unpacked_fence_idx):
    layers = []
    first = unpacked_fence_idx[0]
    bs = first.shape[0]
    batch_dim = torch.arange(bs, device = first.device)[:, None]
    for layer in unpacked_fence_idx:
        full_layer = torch.zeros(bs, layer.max() + 1, dtype = torch.bool, device = first.device)
        full_layer[batch_dim, layer] = True
        layers.append(full_layer)
    return torch.cat(layers, dim = 1)


class MAryPennOperator(PennOperator):
    def __init__(self, model, get_datasets, recorder, i2vs, evalb, train_config):
        super().__init__(model, get_datasets, recorder, i2vs, evalb, train_config)
        self._initial_run = True, True

    def _step(self, mode, ds_name, batch, batch_id = None):

        supervised_signals = {}
        if mode == M_TRAIN:
            supervised_signals['supervised_fence'] = gold_fences = unpack_fence(batch['fence'], batch['segment'], True)
            supervised_signals['keep_low_attention_rate'] = self._train_config.keep_low_attention_rate
        if 'plm_idx' in batch:
            for x in ('plm_idx', 'plm_start'):
                supervised_signals[x] = batch[x]

        batch_time = time()
        (batch_size, batch_len, static, top3_label_logits,
         existences, embeddings, weights, fence_logits, fence_idx, tag_logits, label_logits,
         segment, seg_length) = self._model(batch['token'], self._tune_pre_trained, **supervised_signals)
        batch_time = time() - batch_time

        fences = fence_logits > 0
        if not self._train_config.fence_hinge_loss:
            fence_logits = self._sigmoid(fence_logits)

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, :batch_len]
            tag_mis      = (tags    != batch['tag'])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | bottom_existence)
            label_weight = (label_mis | existences)
            extended_gold_fences = extend_fence_idx(gold_fences)
            
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits)
            if self._train_config.fence_hinge_loss:
                fence_loss = hinge_loss(fence_logits, extended_gold_fences, None)
            else:
                fence_loss = binary_cross_entropy(fence_logits, extended_gold_fences, None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.fence * fence_loss + total_loss
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                                      Tag   = 1 - fraction(tag_mis,     tag_weight),
                                      Label = 1 - fraction(label_mis, label_weight),
                                      Fence = fraction(fences == extended_gold_fences))
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag   = tag_loss,
                                      Label = label_loss,
                                      Fence = fence_loss,
                                      Total = total_loss)
            batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
            if 'segment' in batch:
                batch_kwargs['Height'] = batch['segment'].shape[0]
            self.recorder.tensorboard(self.global_step, 'Batch/%s', **batch_kwargs)
        else:
            vis, _, _ = self._vis_mode

            b_size = (batch_len,)
            b_head = tuple((batch[x].type(torch.uint8) if x in ('tag', 'label', 'fence') else batch[x]).cpu().numpy() for x in 'token tag label fence'.split())
            b_head = b_head + (batch['segment'].cpu().numpy(), batch['seg_length'].cpu().numpy())
            # batch_len, length, token, tag, label, fence, segment, seg_length

            tags   = self._model.get_decision(tag_logits  ).type(torch.uint8).cpu().numpy()
            labels = self._model.get_decision(label_logits).type(torch.uint8).cpu().numpy()
            weight = mean_stdev(weights).cpu().numpy()
            b_data = (tags, labels, fence_idx.type(torch.uint8).cpu().numpy(), weight, segment, seg_length.type(torch.uint8).cpu().numpy())
            # tag, label, fence, segment, seg_length
            tensors = b_size + b_head + b_data
            vis.process(batch_id, tensors)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_bins, test_bins = self._mode_length_bins
        devel_init, test_init = self._initial_run
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            if final_test:
                folder = 'penn_test'
                draw_weights = True
            else:
                folder = 'penn_test_with_devel'
                if int(epoch_minor) == 0:
                    draw_weights = is_bin_times(int(epoch_major))
                else:
                    draw_weights = False
            length_bins = test_bins
            flush_heads = test_init
            self._initial_run = devel_init, False
        else:
            folder = 'penn_devel'
            length_bins = devel_bins
            if int(epoch_minor) == 0:
                draw_weights = is_bin_times(int(epoch_major))
            else:
                draw_weights = False
            flush_heads = devel_init
            self._initial_run = False, test_init

        # if self._model._input_layer.has_static_pca:
        #     self._model._input_layer.flush_pc_if_emb_is_tuned()

        vis = MAryVis(epoch,
                      self.recorder.create_join(folder),
                      self._evalb,
                      self.i2vs,
                      self.recorder.log,
                      draw_weights,
                      length_bins,
                      flush_heads)
        vis = VisRunner(vis, async_ = True) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test = self._vis_mode
        scores, desc, logg = vis.after()
        length_bins = vis.length_bins
        devel_bins, test_bins = self._mode_length_bins
        if length_bins is not None:
            if use_test_set:
                self._mode_length_bins = devel_bins, length_bins # change test
            else:
                self._mode_length_bins = length_bins, test_bins # change devel

            for wbin in length_bins:
                fhead = vis._join_fn(f'head.bin_{wbin}.tree')
                fdata = vis._join_fn(f'data.{self.epoch}.bin_{wbin}.tree')
                if final_test:
                    remove(fhead)
                remove(fdata)

        speed = float(f'{count / seconds:.1f}')
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        logg += f' @{speed}sps. (sym:nn {rate:.2f})'
        scores['speed'] = speed
        if not final_test:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores.get('F1', 0), SamplePerSec = speed)
        self._vis_mode = None
        return scores, desc, logg


from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from data.m_ary import get_tree_from_signals, draw_str_lines
from visualization import tee_trees
from itertools import count

def batch_trees(b_word, b_tag, b_label, b_fence, b_segment, b_seg_length, i2vs, fb_label, b_weight = None):
    for sid, word, tag, label, fence, seg_length in zip(count(), b_word, b_tag, b_label, b_fence, b_seg_length):
        layers_of_label = []
        layers_of_fence = []
        layers_of_weight = None if b_weight is None else []
        label_start = 0
        fence_start = 0
        for l_cnt, l_size, l_len in zip(count(), b_segment, seg_length):
            label_layer = tuple(i2vs.label[i] for i in label[label_start: label_start + l_len])
            layers_of_label.append(label_layer)
            if l_cnt:
                layers_of_fence.append(fence[fence_start: fence_start + l_len + 1])
                fence_start += l_size + 1
            else:
                ln = l_len
            if l_len == 1:
                break
            if b_weight is not None:
                layers_of_weight.append(b_weight[sid, label_start: label_start + l_len])
            label_start += l_size
        wd = [i2vs.token[i] for i in word[:ln]]
        tg = [i2vs.tag  [i] for i in  tag[:ln]]
        yield get_tree_from_signals(wd, tg, layers_of_label, layers_of_fence, fb_label, layers_of_weight)


class MAryVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 draw_weights   = False,
                 length_bins    = None,
                 flush_heads    = False):
        super().__init__(epoch)
        self._evalb = evalb
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._join_fn = lambda x: join(work_dir, x)
        self._is_anew = not isfile(htree) or flush_heads
        self._rpt_file = join(work_dir, f'data.{epoch}.rpt')
        self._logger = logger
        self._fnames = htree, dtree
        self._i2vs = i2vs
        self.register_property('length_bins',  length_bins)
        self._draw_file = join(work_dir, f'data.{epoch}.art') if draw_weights else None
        self._error_idx = 0, []

    def _before(self):
        if self._is_anew:
            self.register_property('length_bins', set())
            for fn in self._fnames:
                if isfile(fn):
                    remove(fn)
        if self._draw_file and isfile(self._draw_file):
            remove(self._draw_file)

    def _process(self, batch_id, batch):
        (batch_len, h_token, h_tag, h_label, h_fence, h_segment, h_seg_length,
         d_tag, d_label, d_fence, d_weight, d_segment, d_seg_length) = batch

        if self._is_anew:
            trees = batch_trees(h_token, h_tag, h_label, h_fence, h_segment, h_seg_length, self._i2vs, None)
            trees = [' '.join(str(x).split()) for x in trees]
            self.length_bins |= tee_trees(self._join_fn, 'head', h_seg_length[:, 0], trees, None, 10)

        trees = []
        idx_cnt, error_idx = self._error_idx
        for tree, safe in batch_trees(h_token, d_tag, d_label, d_fence, d_segment, d_seg_length, self._i2vs, 'S'):
            idx_cnt += 1 # start from 1
            if not safe:
                error_idx.append(idx_cnt)
            trees.append(' '.join(str(tree).split()))
        self._error_idx = idx_cnt, error_idx

        if self._draw_file is None:
            bin_size = None
        else:
            bin_size = None if self.length_bins is None else 10
            with open(self._draw_file, 'a+') as fw:
                for tree, safe in batch_trees(h_token, d_tag, d_label, d_fence, d_segment, d_seg_length, self._i2vs, 'S', d_weight):
                    if not safe:
                        fw.write('\n[FORCING TREE WITH ROOT = S]\n')
                    try:
                        fw.write('\n'.join(draw_str_lines(tree)) + '\n\n')
                    except:
                        fw.write('FAILING DRAWING\n\n')
        tee_trees(self._join_fn, f'data.{self.epoch}', d_seg_length[:, 0], trees, None, bin_size)

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        proc = parseval(self._evalb, *self._fnames)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        idx_cnt, error_idx = self._error_idx
        error_cnt = len(error_idx)
        if error_cnt < 50:
            self._logger(f'  {error_cnt} conversion errors')
        else:
            self._logger(f'  {error_cnt} conversion errors: ' + ' '.join(str(x) for x in error_idx))
        if num_errors:
            self._logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    self._logger(f'    {e}. ' + error)
        with open(self._rpt_file, 'w') as fw:
            fw.write(report)
            if num_errors >= 10:
                self._logger(f'  Go check {self._rpt_file} for details.')
                fw.write('\n\n' + '\n'.join(errors))

        if self.length_bins is not None and self._draw_file is not None:
            with open(self._join_fn(f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self.length_bins:
                    fhead = self._join_fn(f'head.bin_{wbin}.tree')
                    fdata = self._join_fn(f'data.{self.epoch}.bin_{wbin}.tree')
                    proc = parseval(self._evalb, fhead, fdata)
                    smy = rpt_summary(proc.stdout.decode(), False, True)
                    fw.write(f"{wbin},{smy['N']},{smy['LP']},{smy['LR']},{smy['F1']},{smy['TA']}\n")

        desc = f'Evalb({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger
