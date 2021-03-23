import torch
from torch import nn
from utils.operator import Operator
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, false_type, tune_epoch_type, frac_06, frac_close
from models.utils import PCA, fraction, hinge_score, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.t_lstm_dccp.operator import DiscoOperator, DiscoVis, inner_score
from utils.shell_io import has_discodop, discodop_eval, byte_style

train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     fence = BaseType(0.5, validator = frac_open_0),
                                     disco_1d = BaseType(0.5, validator = frac_open_0),
                                     disco_2d = BaseType(0.5, validator = frac_open_0)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained_from_nth_epoch = tune_epoch_type,
                  lr_factor_for_tuning = frac_06,
                  multiprocessing_decode = false_type,
                  binary_hinge_loss = true_type)


class DiscoMultiOperator(DiscoOperator):
    def _step(self, mode, ds_name, batch, batch_id = None):

        supervised_signals = {}
        if mode == M_TRAIN:
            all_space = batch['space']
            dis_disco = batch['dis_disco']
            dis_start = con_start = 0
            space_layers = []
            disco_layers = []
            for dis_seg in batch['segment']:
                dis_end = dis_start + dis_seg
                space_layers.append(all_space[:, dis_start:dis_end])
                disco_layers.append(dis_disco[:, dis_start:dis_end])
                dis_start = dis_end
            # for con_seg in batch['split_segment']:
            #     con_end = con_start + con_seg
            #     print(con_split[:, con_start:con_end] * 1)
            #     con_start = con_end
            # dis_comp = batch['dis_component']
            # dis_shape = batch['dis_shape']
            # dis_slice = batch['dis_slice']
            # print(dis_comp.shape)
            # for start, end, shape in zip(dis_slice, dis_slice[1:], dis_shape):
            #     if start < end:
            #         comp = dis_comp[start:end].reshape(shape)
            #         print(comp * 1)
            supervised_signals['supervision'] = space_layers, disco_layers
            # if 'dis_' batch
        if 'plm_idx' in batch:
            for x in ('plm_idx', 'plm_start'):
                supervised_signals[x] = batch[x]
        has_disco_2d = 'dis_component' in batch

        batch_time = time()
        (batch_size, batch_len, static, top3_label_logits,
         existences, embeddings, weights, disco_1d_logits, fence_logits, disco_2d_logits, space, tag_logits, label_logits,
         segment, seg_length) = self._model(batch['token'], self._tune_pre_trained, **supervised_signals)
        batch_time = time() - batch_time

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, :batch_len]
            tag_mis      = (tags    != batch['tag'])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | bottom_existence)
            label_weight = (label_mis | existences)
            gold_fences  = batch['con_split']

            fences = fence_logits > 0
            disco_1d = disco_1d_logits > 0
            if has_disco_2d:
                disco_2d = disco_2d_logits > 0
            else:
                assert disco_2d_logits is None
            if not self._train_config.binary_hinge_loss:
                fence_logits = self._sigmoid(fence_logits)
                disco_1d_logits = self._sigmoid(disco_1d_logits)
                if has_disco_2d:
                    disco_2d_logits = self._sigmoid(disco_2d_logits)

            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits)
            if self._train_config.binary_hinge_loss:
                fence_loss = hinge_loss(fence_logits, gold_fences, None)
                disco_1d_loss = hinge_loss(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = hinge_loss(disco_2d_logits, batch['dis_component'], None)
            else:
                fence_loss = binary_cross_entropy(fence_logits, gold_fences, None)
                disco_1d_loss = binary_cross_entropy(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = binary_cross_entropy(disco_2d_logits, batch['dis_component'], None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.fence * fence_loss + total_loss
            total_loss = self._train_config.loss_weight.disco_1d * disco_1d_loss + total_loss
            if has_disco_2d:
                total_loss = self._train_config.loss_weight.disco_2d * disco_2d_loss + total_loss
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                                      Tag   = 1 - fraction(tag_mis,     tag_weight),
                                      Label = 1 - fraction(label_mis, label_weight),
                                      Fence = fraction(fences == gold_fences),
                                      Disco_1D = fraction(disco_1d == batch['dis_disco']),
                                      Disco_2D = fraction(disco_2d == batch['dis_component']) if has_disco_2d else None)
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag   = tag_loss,
                                      Label = label_loss,
                                      Fence = fence_loss,
                                      Disco_1D = disco_1d_loss,
                                      Disco_2D = disco_2d_loss if has_disco_2d else None,
                                      Total = total_loss)
            batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
            if 'segment' in batch:
                batch_kwargs['Height'] = batch['segment'].shape[0]
            self.recorder.tensorboard(self.global_step, 'Batch/%s', **batch_kwargs)
        else:
            vis, _, _, serial = self._vis_mode

            tags   = self._model.get_decision(tag_logits  ).type(torch.uint8).cpu().numpy()
            labels = self._model.get_decision(label_logits).type(torch.uint8).cpu().numpy()
            spaces = (space - 1).type(torch.int16).cpu().numpy() # internal diff: data/model
            seg_length = seg_length.type(torch.int16).cpu().numpy()
            if serial:
                b_size = (batch_len,)
                b_head = []
                for fn in ('token', 'tag', 'label', 'space', 'segment', 'seg_length'):
                    tensor = batch[fn]
                    if fn == 'space':
                        tensor -= 1
                    if fn in ('tag', 'label', 'space'):
                        tensor = tensor.type(torch.uint8)
                    b_head.append(tensor.cpu().numpy())
                b_head = tuple(b_head)
                weight = mean_stdev(weights).cpu().numpy()
                b_data = (tags, labels, spaces, weight, segment, seg_length)
            else:
                b_size = (batch_len,)
                b_head = (batch['token'].cpu().numpy(),)
                # batch_size, segment, token, tag, label, fence, seg_length
                b_data = (tags, labels, spaces, segment, seg_length)

            # tag, label, fence, segment, seg_length
            tensors = b_size + b_head + b_data
            vis.process(batch_id, tensors)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        # devel_bins, test_bins = self._mode_length_bins
        devel_head_batchess, test_head_batchess = self._mode_trees
        if use_test_set:
            head_trees = test_head_batchess
            if final_test:
                folder = ds_name + '_test'
                draw_trees = True
            else:
                folder = ds_name + '_test_with_devel'
                draw_trees = is_bin_times(int(float(epoch)) - 1)
        else:
            head_trees = devel_head_batchess
            folder = ds_name + '_devel'
            draw_trees = is_bin_times(int(float(epoch)) - 1)
        if self._optuna_mode:
            draw_trees = False
        vis = DiscoMultiVis(epoch,
                            self.recorder.create_join(folder),
                            self.i2vs,
                            head_trees,
                            self.recorder.log,
                            self._evalb_lcfrs_kwargs,
                            self._discodop_prm,
                            draw_trees)
        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = False) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads


from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from data.cross.multib import disco_tree, draw_str_lines
from visualization import tee_trees
from itertools import count

def batch_trees(b_word, b_tag, b_label, b_space, b_segment, b_seg_length, i2vs, fb_label = None, b_weight = None):
    for sid, word, tag, label, space, seg_length in zip(count(), b_word, b_tag, b_label, b_space, b_seg_length):
        layers_of_label = []
        layers_of_space = []
        layers_of_weight = None if b_weight is None else []
        label_start = 0
        for l_size, l_len in zip(b_segment, seg_length):
            label_end = label_start + l_len
            label_layer = label[label_start: label_end]
            layers_of_label.append(tuple(i2vs.label[i] for i in label_layer))
            if l_len == 1:
                break
            layers_of_space.append(space[label_start: label_end])
            if b_weight is not None:
                layers_of_weight.append(b_weight[sid, label_start: label_end])
            label_start += l_size
        ln = seg_length[0]
        wd = [i2vs.token[i] for i in word[:ln]]
        tg = [i2vs.tag  [i] for i in  tag[:ln]]
        yield disco_tree(wd, tg, layers_of_label, layers_of_space, fb_label) # layers_of_weight


from data.cross.evalb_lcfrs import DiscoEvalb, ExportWriter, read_param
class DiscoMultiVis(DiscoVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, draw_trees):
        super().__init__(epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, None, False)
        self._draw_trees = self._dtv.join(f'tree.{epoch}.art') if draw_trees else None

    def _process(self, batch_id, batch):
        bid_offset, _ = self._evalb.total_missing

        i2vs = self._dtv.vocabs
        if self._pending_heads:
            (batch_len, h_token, h_tag, h_label, h_space, h_segment, h_seg_length,
             d_tag, d_label, d_space, d_weight, d_segment, d_seg_length) = batch
            head_lines = []
            head_trees_for_scores = []
            for btm, td, rt, error in batch_trees(h_token, h_tag, h_label, h_space, h_segment, h_seg_length, i2vs):
                assert not error
                head_lines.append('\n'.join(draw_str_lines(btm, td, root_stamp = ' (Gold)')))
                head_trees_for_scores.append(inner_score(btm, td, rt, self._xh_writer, **self._evalb_lcfrs_kwargs))
            self._head_batches.append((head_trees_for_scores, head_lines))
        else:
            (batch_len, h_token, d_tag, d_label, d_space, d_segment, d_seg_length) = batch
            head_trees_for_scores, head_lines = self._head_batches[self._data_batch_cnt]
            self._data_batch_cnt += 1
        
        # data_errors = []
        lines = ''
        self._evalb.add_batch_line(batch_id)
        for sid, (btm, td, rt, error) in enumerate(batch_trees(h_token, d_tag, d_label, d_space, d_segment, d_seg_length, i2vs, 'VROOT')):
            pred_gold = inner_score(btm, td, rt, self._xd_writer, **self._evalb_lcfrs_kwargs) + head_trees_for_scores[sid]
            bracket_match, p_num_brackets, g_num_brackets, dbm, pdbc, gdbc, tag_match, g_tag_count = self._evalb.add(*pred_gold)
            if error: self._v_errors[bid_offset + sid] = error
            if self._draw_trees:
                lines += f'Batch #{batch_id} ───────────────────────────────────────────\n'
                lines += f'  Sent #{sid} | #{bid_offset + sid}: '
                tag_line = 'Exact Tagging Match' if tag_match == g_tag_count else f'Tagging: {tag_match}/{g_tag_count}'
                if pdbc or gdbc:
                    tag_line += ' | DISC.'
                    if not dbm and gdbc:
                        tag_line += ' failed'
                    if not gdbc and pdbc:
                        tag_line += ' overdone'
                if bracket_match == g_num_brackets:
                    if tag_match == g_tag_count:
                        lines += 'Exact Match\n\n'
                    else:
                        lines += 'Exact Bracketing Match | ' + tag_line + '\n\n'
                    lines += head_lines[sid]
                else:
                    lines += f'Bracketing {p_num_brackets} > {bracket_match} < {g_num_brackets} | '
                    lines += tag_line + '\n\n'
                    lines += head_lines[sid] + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, root_stamp = ' (Predicted)'))
                lines += '\n\n\n'
        if self._draw_trees:
            with open(self._draw_trees, 'a+') as fw:
                fw.write(lines)