import torch
from torch import nn
from utils.operator import Operator
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06, frac_close
from models.utils import PCA, fraction, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.t_lstm_dccp.operator import DiscoOperator, DiscoVis, inner_score, ParallelVis as BinaryParallelVis

train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     fence = BaseType(0.5, validator = frac_open_0),
                                     disco_1d = BaseType(0.5, validator = frac_open_0),
                                     disco_2d = BaseType(0.5, validator = frac_open_0),
                                     disco_2d_neg = BaseType(0.5, validator = frac_open_0)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  disco_2d_negrate = BaseType(0.05, validator = frac_close),
                  binary_hinge_loss = true_type,
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))


class DiscoMultiOperator(DiscoOperator):
    def _step(self, mode, ds_name, batch, batch_id = None):

        supervised_signals = {}
        has_disco_2d = 'dis_component' in batch
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
            if self._train_config.disco_2d_negrate:
                supervised_signals['disco_2d_negative'] = self._train_config.disco_2d_negrate
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
        if 'sub_idx' in batch:
            supervised_signals['sub_idx'] = batch['sub_idx']
        if 'sub_fence' in batch:
            supervised_signals['sub_fence'] = batch['sub_fence']
        elif 'plm_idx' in batch:
            for x in ('plm_idx', 'plm_start'):
                supervised_signals[x] = batch[x]

        batch_time = time()
        (batch_size, batch_len, static, top3_label_logits,
         existences, embeddings, weights, disco_1d_logits, fence_logits, disco_2d_logits, disco_2d_negative, space, tag_logits, label_logits,
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
                _sigmoid = self._model._sigmoid
                fence_logits = _sigmoid(fence_logits)
                disco_1d_logits = _sigmoid(disco_1d_logits)
                if has_disco_2d:
                    disco_2d_logits = _sigmoid(disco_2d_logits)
            
            has_disco_2d_negative = disco_2d_negative is not None
            if has_disco_2d_negative:
                disco_2d_negative_accuracy = disco_2d_negative < 0
                if not self._train_config.binary_hinge_loss:
                    disco_2d_negative = _sigmoid(disco_2d_negative)

            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits)
            if self._train_config.binary_hinge_loss:
                fence_loss = hinge_loss(fence_logits, gold_fences, None)
                disco_1d_loss = hinge_loss(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = hinge_loss(disco_2d_logits, batch['dis_component'], None)
                    if has_disco_2d_negative:
                        disco_2d_negloss = hinge_loss(disco_2d_negative, torch.zeros_like(disco_2d_negative, dtype = torch.bool), None)
            else:
                fence_loss = binary_cross_entropy(fence_logits, gold_fences, None)
                disco_1d_loss = binary_cross_entropy(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = binary_cross_entropy(disco_2d_logits, batch['dis_component'], None)
                    if has_disco_2d_negative:
                        disco_2d_negloss = binary_cross_entropy(disco_2d_negative, torch.zeros_like(disco_2d_negative, dtype = torch.bool), None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.fence * fence_loss + total_loss
            total_loss = self._train_config.loss_weight.disco_1d * disco_1d_loss + total_loss
            if has_disco_2d:
                total_loss = self._train_config.loss_weight.disco_2d * disco_2d_loss + total_loss
                if has_disco_2d_negative:
                    total_loss = self._train_config.loss_weight.disco_2d_neg * disco_2d_negloss + total_loss

            total_loss.backward()
            
            if hasattr(self._model, 'tensorboard'):
                self._model.tensorboard(self.recorder, self.global_step)
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                                      Tag   = 1 - fraction(tag_mis,     tag_weight),
                                      Label = 1 - fraction(label_mis, label_weight),
                                      Fence = fraction(fences == gold_fences),
                                      Disco_1D = fraction(disco_1d == batch['dis_disco']),
                                      Disco_2D = fraction(disco_2d == batch['dis_component']) if has_disco_2d else None,
                                      Disco_2D_Neg = fraction(disco_2d_negative_accuracy) if has_disco_2d_negative else None)
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag   = tag_loss,
                                      Label = label_loss,
                                      Fence = fence_loss,
                                      Disco_1D = disco_1d_loss,
                                      Disco_2D = disco_2d_loss if has_disco_2d else None,
                                      Disco_2D_Neg = disco_2d_negloss if has_disco_2d_negative else None,
                                      Total = total_loss)
            batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
            if 'segment' in batch:
                batch_kwargs['Height'] = batch['segment'].shape[0]
            self.recorder.tensorboard(self.global_step, 'Batch/%s', **batch_kwargs)
        else:
            vis, _, _, pending_heads, _ = self._vis_mode

            tags   = self._model.get_decision(tag_logits  ).type(torch.uint8).cpu().numpy()
            labels = self._model.get_decision(label_logits).type(torch.uint8).cpu().numpy()
            seg_length = seg_length.type(torch.int16).cpu().numpy()
            spaces = (space - 1).type(torch.int16).cpu().numpy() # internal diff: data/model

            if vis._draw_trees:
                _sigmoid = self._model._sigmoid
                fence    = _sigmoid(fence_logits)
                disco_1d = _sigmoid(disco_1d_logits)
                weight   = mean_stdev(weights).type(torch.float16).cpu().numpy()
                disco_2d = None if disco_2d_logits is None else disco_2d_logits
            else:
                weight = fence = disco_1d = disco_2d = None

            if pending_heads:
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
                b_data = (tags, labels, spaces, weight, disco_2d_logits, segment, seg_length)
            else:
                b_size = (batch_len,)
                b_head = (batch['token'].cpu().numpy(),)
                b_data = (tags, labels, spaces, weight, disco_2d_logits, segment, seg_length)

            # tag, label, fence, segment, seg_length
            tensors = b_size + b_head + b_data
            vis.process(batch_id, tensors)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        # devel_bins, test_bins = self._mode_length_bins
        devel_head_batchess, test_head_batchess = self._mode_trees
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            head_trees = test_head_batchess
            if final_test:
                folder = ds_name + '_test'
                draw_trees = True
            else:
                folder = ds_name + '_test_with_devel'
                draw_trees = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
        else:
            head_trees = devel_head_batchess
            folder = ds_name + '_devel'
            draw_trees = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
        if self._optuna_mode:
            draw_trees = False
        work_dir = self.recorder.create_join(folder)
        serial = draw_trees or not head_trees or self.dm is None
        if serial:
            async_ = True
            vis = DiscoMultiVis(epoch,
                                work_dir,
                                self.i2vs,
                                head_trees,
                                self.recorder.log,
                                self._evalb_lcfrs_kwargs,
                                self._discodop_prm,
                                draw_trees)
        else:
            async_ = False
            vis = ParallelVis(epoch, work_dir, self.i2vs, self.recorder.log, self._evalb_lcfrs_kwargs, self._discodop_prm, self.dm)
        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = async_) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads, serial

    def _get_optuna_fn(self, train_params):
        import numpy as np
        from utils.train_ops import train, get_optuna_params
        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                loss_weight = specs['train']['loss_weight']
                loss_weight['tag']   = t = trial.suggest_float('tag',   0.0, 1.0)
                loss_weight['label'] = l = trial.suggest_float('label', 0.0, 1.0)
                loss_weight['fence'] = f = trial.suggest_float('fence', 0.0, 1.0)
                loss_weight['disco_1d'] = d1 = trial.suggest_float('disco_1d', 0.0, 1.0)
                loss_weight['disco_2d'] = d2 = trial.suggest_float('disco_2d', 0.0, 1.0)
                loss_str = 'L=' + height_ratio(t) + height_ratio(l) + height_ratio(f) + height_ratio(d1) + height_ratio(d2)

                if d_d2n := specs['train']['disco_2d_negrate'] > 0:
                    specs['train']['disco_2d_negrate'] = d_d2n = trial.suggest_loguniform('disco_2d_negrate', 0.0001, 0.1)
                    loss_weight['disco_2d_neg'] = l_d2n = trial.suggest_float('disco_2d_neg', 1e-6, 1.0)
                    loss_str += 'N' + height_ratio(l_d2n)
                else:
                    loss_weight['disco_2d_neg'] = specs['train']['disco_2d_negrate'] = 0

                medium_factor = specs['data']
                medium_factor = specs['data'][get_sole_key(medium_factor)]['medium_factor']

                if involve_balanced := medium_factor['balanced'] > 0:
                    medium_factor['balanced'] = bz = trial.suggest_float('balanced', 0.0, 1.0)

                if multi_medoid := any(0 < v < 1 for v in medium_factor['others'].values()):
                    med_k, med_v = [], []
                    for k, v in medium_factor['others'].items():
                        if v > 0:
                            med_k.append(k)
                            med_v.append(trial.suggest_loguniform(k, 1e-6, 1e3))
                    med_v = np.array(med_v)
                    med_v /= np.sum(med_v)
                    for k, v in zip(med_k, med_v):
                        medium_factor['others'][k] = float(v)

                self._train_materials = medium_factor, self._train_materials[1]
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                self._train_config._nested.update(specs['train'])
                med_str = 'med='
                if involve_balanced: med_str += f'{height_ratio(bz)}'
                if multi_medoid:     med_str += 'M' + ''.join(height_ratio(v) for v in med_v)
                if d_d2n > 0:        med_str += f'Ng={d_d2n:.1e}'
                return med_str + ';' + loss_str + f';lr={lr:.1e}'

            self._mode_trees = [], [] # force init
            self.setup_optuna_mode(spec_update_fn, trial)

            return train(optuna_params, self)['key']
        return obj_fn

from utils.vis import VisRunner
from utils.file_io import remove, isdir, mkdir, listdir, join
from data.cross.multib import disco_tree, draw_str_lines
from itertools import count

def batch_trees(b_word, b_tag, b_label, b_space, b_segment, b_seg_length, i2vs, fb_label = None, b_weight = None):
    add_weight = b_weight is not None
    for sid, word, tag, label, space, seg_length in zip(count(), b_word, b_tag, b_label, b_space, b_seg_length):
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
        yield disco_tree(wd, tg, layers_of_label, layers_of_space, fb_label, layers_of_weight)


from utils.str_ops import cat_lines, height_ratio, space_height_ratio
class DiscoMultiVis(DiscoVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, draw_trees):
        super().__init__(epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, None, False)
        if draw_trees:
            draw_trees = self._dtv.join(f'tree.{epoch}.art')
            if isdir(draw_trees):
                for fname in listdir(draw_trees):
                    remove(join(draw_trees, fname))
            else:
                mkdir(draw_trees)
        self._draw_trees = draw_trees

    def _process(self, batch_id, batch):
        bid_offset, _ = self._evalb.total_missing

        i2vs = self._dtv.vocabs
        if self._pending_heads:
            (batch_len, h_token, h_tag, h_label, h_space, h_segment, h_seg_length,
             d_tag, d_label, d_space, d_weight, d_disco_2d, d_segment, d_seg_length) = batch
            head_lines = []
            head_trees_for_scores = []
            for btm, td, rt, error in batch_trees(h_token, h_tag, h_label, h_space, h_segment, h_seg_length, i2vs):
                assert not error
                head_lines.append('\n'.join(draw_str_lines(btm, td, attachment = ' (Gold)')))
                head_trees_for_scores.append(inner_score(btm, td, rt, self._xh_writer, **self._evalb_lcfrs_kwargs))
            self._head_batches.append((head_trees_for_scores, head_lines))
        else:
            (batch_len, h_token, d_tag, d_label, d_space, d_weight, d_disco_2d, d_segment, d_seg_length) = batch
            head_trees_for_scores, head_lines = self._head_batches[self._data_batch_cnt]
            self._data_batch_cnt += 1
        
        # data_errors = []
        self._evalb.add_batch_line(batch_id)
        evalb_lines = []
        for sid, (btm, td, rt, error) in enumerate(batch_trees(h_token, d_tag, d_label, d_space, d_segment, d_seg_length, i2vs, 'VROOT')):
            pred_gold = inner_score(btm, td, rt, self._xd_writer, **self._evalb_lcfrs_kwargs) + head_trees_for_scores[sid]
            if self._draw_trees: evalb_lines.append(self._evalb.add(*pred_gold))
            if error: self._v_errors[bid_offset + sid] = error
        if self._draw_trees:
            has_disco_2d = d_disco_2d is not None and any(l for l in d_disco_2d)
            c_lines = d_lines = m_lines = f'Batch #{batch_id} ───────────────────────────────────────────\n'
            c_cnt = d_cnt = m_cnt = 0
            d_has_n_comps = d_has_n_fallback = m_has_n_comps = m_has_n_fallback = 0
            for sid, (btm, td, rt, error, wns) in enumerate(batch_trees(h_token, d_tag, d_label, d_space, d_segment, d_seg_length, i2vs, 'VROOT', d_weight)):
                bracket_match, p_num_brackets, g_num_brackets, dbm, pdbc, gdbc, tag_match, g_tag_count = evalb_lines[sid]
                lines = f'Sent #{sid} | #{bid_offset + sid}: '
                tag_line = 'Exact Tagging Match' if tag_match == g_tag_count else f'Tagging: {tag_match}/{g_tag_count}'
                if pdbc or gdbc:
                    tag_line += ' | DISC.'
                    if not dbm and gdbc:
                        tag_line += ' failed'
                    if not gdbc and pdbc:
                        tag_line += ' overdone'

                has_n_comps = has_n_fallback = 0
                if has_disco_2d and any(sid in l for l in d_disco_2d):
                    base_lines = ['2D ']
                    base_lines += [''] * max(len(l[sid][0]) for l in d_disco_2d if sid in l)
                    for lid, l_disco_2d in enumerate(d_disco_2d):
                        if sid in l_disco_2d:
                            new_lines = f'#{lid + 1}'
                            mat, _, _, n_comps, fallback, thresh, n_trial = l_disco_2d[sid]
                            if fallback:
                                new_lines += '**'
                                has_n_fallback += 1
                            else:
                                if n_comps > 1:
                                    new_lines += f'*{n_comps}'
                                    has_n_comps = max(has_n_comps, n_comps)
                                if thresh != 0.5:
                                    new_lines += f'!{thresh:0.2f}'
                            if n_trial > 1:
                                new_lines += f'@{n_trial}'
                            new_lines = [new_lines]
                            for row in mat:
                                new_lines.append(''.join(space_height_ratio(val) for val in row) + ' | ')
                            base_lines = cat_lines(base_lines, new_lines)
                    disco_2d_lines = '\n\n' + '\n'.join(base_lines)
                else:
                    disco_2d_lines = ''

                heights = {}
                for nid, height in wns.items(): # weight_nodes
                    if nid != rt:
                        heights[nid] = f'#{height}'
                    else:
                        heights[nid] = f'#{height} (Predicted)'
                if rt not in heights:
                    heights[rt] = ' (Predicted)'

                if bracket_match == g_num_brackets:
                    if tag_match == g_tag_count:
                        lines += 'Exact Match\n\n'
                    else:
                        lines += 'Exact Bracketing Match | ' + tag_line + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, attachment = heights))
                    m_lines += lines + disco_2d_lines + '\n\n\n'
                    m_cnt += 1
                    m_has_n_fallback = max(m_has_n_fallback, has_n_fallback)
                    m_has_n_comps = max(m_has_n_comps, has_n_comps)
                else:
                    lines += f'Bracketing {p_num_brackets} > {bracket_match} < {g_num_brackets} | '
                    lines += tag_line + '\n\n'
                    lines += head_lines[sid] + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, attachment = heights))
                    if pdbc or gdbc:
                        d_lines += lines + disco_2d_lines + '\n\n\n'
                        d_cnt += 1
                        d_has_n_fallback = max(d_has_n_fallback, has_n_fallback)
                        d_has_n_comps = max(d_has_n_comps, has_n_comps)
                    else:
                        c_lines += lines + disco_2d_lines + '\n\n\n'
                        c_cnt += 1
            fname_prefix = join(self._draw_trees, f'{batch_id:03d}.')
            total = c_cnt + d_cnt + m_cnt
            for suffix, lines, cnt in zip('cdm', (c_lines, d_lines, m_lines), (c_cnt, d_cnt, m_cnt)):
                if cnt > 0:
                    if suffix == 'd':
                        if d_has_n_fallback:
                            suffix += '.fb' if d_has_n_fallback < 2 else f'.{d_has_n_fallback}fb'
                        if d_has_n_comps:
                            suffix += '.comps' if d_has_n_comps < 3 else f'.{d_has_n_comps}comps'
                    if suffix == 'm':
                        if m_has_n_fallback:
                            suffix += '.fb' if m_has_n_fallback < 2 else f'.{m_has_n_fallback}fb'
                        if m_has_n_comps:
                            suffix += '.comps' if m_has_n_comps < 3 else f'.{m_has_n_comps}comps'
                    suffix = '.' + suffix + '.art'
                    with open(fname_prefix + f'{height_ratio(cnt / total)}' + suffix, 'w') as fw:
                        fw.write(lines)


class ParallelVis(BinaryParallelVis):
    _draw_trees = False
    
    def _process(self, batch_id, batch):
        (batch_len, h_token, d_tag, d_label, d_space, d_weight, d_disco_2d, d_segment, d_seg_length) = batch
        self._args[0].batch(batch_id, self._bid_offset, d_segment, d_seg_length, h_token, d_tag, d_label, d_space)
        self._bid_offset += h_token.shape[0]