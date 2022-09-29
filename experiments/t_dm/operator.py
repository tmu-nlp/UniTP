import torch
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06, frac_close
from models.utils import fraction, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from data.cross.mp import DVA, DVP, inner_score, m_batch_trees as batch_trees
from experiments.helper import make_tensors
from experiments.helper.do import DO

train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     chunk = BaseType(0.5, validator = frac_open_0),
                                     disco_1d = BaseType(0.5, validator = frac_open_0),
                                     disco_2d = BaseType(0.5, validator = frac_open_0),
                                     disco_2d_intra = BaseType(0.5, validator = frac_close),
                                     disco_2d_inter = BaseType(0.5, validator = frac_close)),
                  learning_rate      = BaseType(0.001, validator = frac_open_0),
                  binary_hinge_loss = true_type,
                  tune_pre_trained  = dict(from_nth_epoch = tune_epoch_type,
                                           lr_factor      = frac_06))


class DMOperater(DO):
    def _step(self, mode, ds_name, batch, batch_id = None):

        batch['key'] = corp = ds_name if self.multi_corp else None
        has_disco_2d = 'dis_component' in batch
        if mode == M_TRAIN:
            all_space = batch['space']
            dis_disco = batch['dis_disco']
            dis_start = 0
            space_layers = []
            disco_layers = []
            for dis_seg in batch['batch_segment']:
                dis_end = dis_start + dis_seg
                space_layers.append(all_space[:, dis_start:dis_end])
                disco_layers.append(dis_disco[:, dis_start:dis_end])
                dis_start = dis_end
            batch['supervision'] = space_layers, disco_layers, batch.get('inter_disco')
        else:
            if (vis := self._vis_mode[0])._draw_trees:
                batch['get_disco_2d'] = True


        batch_time = time()
        bottom, stem, tag_label = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time
        batch_size, batch_len = bottom[:2]
        existences = stem.existence
        (weights, disco_1d_logits, chunk_logits, disco_2d_logits,
         disco_2d_positive, disco_2d_negative, inter_2d_negative, space,
         segment) = stem.extension
        tag_logits, label_logits = tag_label[2:4]

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, :batch_len]
            tag_mis      = (tags    != batch['tag'])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | bottom_existence)
            label_weight = (label_mis | existences)
            gold_chunks  = batch['con_split']

            chunks = chunk_logits > 0
            disco_1d = disco_1d_logits > 0
            if has_disco_2d:
                disco_2d = disco_2d_logits > 0
            else:
                assert disco_2d_logits is None
            if not self._train_config.binary_hinge_loss:
                _sigmoid = self._model.stem._sigmoid
                chunk_logits = _sigmoid(chunk_logits)
                disco_1d_logits = _sigmoid(disco_1d_logits)
                if has_disco_2d:
                    disco_2d_logits = _sigmoid(disco_2d_logits)
            
            if has_disco_2d_positive := (disco_2d_positive is not None):
                disco_2d_positive_accuracy = disco_2d_positive > 0
                if not self._train_config.binary_hinge_loss:
                    disco_2d_positive = _sigmoid(disco_2d_positive)
            
            if has_disco_2d_negative := (disco_2d_negative is not None):
                disco_2d_negative_accuracy = disco_2d_negative <= 0
                if not self._train_config.binary_hinge_loss:
                    disco_2d_negative = _sigmoid(disco_2d_negative)
            
            if has_inter_2d_negative := (inter_2d_negative is not None):
                inter_2d_negative_accuracy = inter_2d_negative <= 0
                if not self._train_config.binary_hinge_loss:
                    inter_2d_negative = _sigmoid(inter_2d_negative)
                    
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, None, corp)
            if self._train_config.binary_hinge_loss:
                chunk_loss = hinge_loss(chunk_logits, gold_chunks, None)
                disco_1d_loss = hinge_loss(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = hinge_loss(disco_2d_logits, batch['dis_component'], None)
                if has_disco_2d_positive:
                    disco_2d_posloss = hinge_loss(disco_2d_positive, torch.ones_like(disco_2d_positive, dtype = torch.bool), None)
                if has_disco_2d_negative:
                    disco_2d_negloss = hinge_loss(disco_2d_negative, torch.zeros_like(disco_2d_negative, dtype = torch.bool), None)
                if has_inter_2d_negative:
                    inter_2d_negloss = hinge_loss(inter_2d_negative, torch.zeros_like(inter_2d_negative, dtype = torch.bool), None)
            else:
                chunk_loss = binary_cross_entropy(chunk_logits, gold_chunks, None)
                disco_1d_loss = binary_cross_entropy(disco_1d_logits, batch['dis_disco'], None)
                if has_disco_2d:
                    disco_2d_loss = binary_cross_entropy(disco_2d_logits, batch['dis_component'], None)
                if has_disco_2d_positive:
                    disco_2d_posloss = binary_cross_entropy(disco_2d_positive, torch.ones_like(disco_2d_positive, dtype = torch.bool), None)
                if has_disco_2d_negative:
                    disco_2d_negloss = binary_cross_entropy(disco_2d_negative, torch.zeros_like(disco_2d_negative, dtype = torch.bool), None)
                if has_inter_2d_negative:
                    inter_2d_negloss = binary_cross_entropy(inter_2d_negative, torch.zeros_like(inter_2d_negative, dtype = torch.bool), None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.chunk * chunk_loss + total_loss
            total_loss = self._train_config.loss_weight.disco_1d * disco_1d_loss + total_loss
            if has_disco_2d:
                total_loss = self._train_config.loss_weight.disco_2d * disco_2d_loss + total_loss
            if has_disco_2d_positive:
                total_loss = self._train_config.loss_weight.disco_2d_intra * disco_2d_posloss + total_loss
            if has_disco_2d_negative:
                total_loss = self._train_config.loss_weight.disco_2d_intra * disco_2d_negloss + total_loss
            if has_inter_2d_negative:
                total_loss = self._train_config.loss_weight.disco_2d_inter * inter_2d_negloss + total_loss
            total_loss.backward()
            
            if self.recorder._writer is not None:
                suffix = ds_name if self.multi_corp else None
                self.recorder.tensorboard(self.global_step, 'Accuracy/%s', suffix,
                    Tag   = 1 - fraction(tag_mis,     tag_weight),
                    Label = 1 - fraction(label_mis, label_weight),
                    Fence = fraction(chunks == gold_chunks),
                    Disco_1D = fraction(disco_1d == batch['dis_disco']),
                    Disco_2D = fraction(disco_2d == batch['dis_component']) if has_disco_2d else None,
                    Disco_2D_Intra_P = fraction(disco_2d_positive_accuracy) if has_disco_2d_positive else None,
                    Disco_2D_Intra_N = fraction(disco_2d_negative_accuracy) if has_disco_2d_negative else None,
                    Disco_2D_Inter_N = fraction(inter_2d_negative_accuracy) if has_inter_2d_negative else None)
                self.recorder.tensorboard(self.global_step, 'Loss/%s', suffix,
                    Tag   = tag_loss,
                    Label = label_loss,
                    Fence = chunk_loss,
                    Disco_1D = disco_1d_loss,
                    Disco_2D = disco_2d_loss if has_disco_2d else None,
                    Disco_2D_Intra_P = disco_2d_posloss if has_disco_2d_positive else None,
                    Disco_2D_Intra_N = disco_2d_negloss if has_disco_2d_negative else None,
                    Disco_2D_Inter_N = inter_2d_negloss if has_inter_2d_negative else None,
                    Total = total_loss)
                batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
                if 'batch_segment' in batch:
                    batch_kwargs['Height'] = batch['batch_segment'].shape[0]
                if has_disco_2d_positive:
                    batch_kwargs['Disco_2D_Intra_P'] = disco_2d_positive_accuracy.shape[0]
                if has_disco_2d_negative:
                    batch_kwargs['Disco_2D_Intra_N'] = disco_2d_negative_accuracy.shape[0]
                if has_inter_2d_negative:
                    batch_kwargs['Disco_2D_Inter_N'] = inter_2d_negative_accuracy.shape[0]
                self.recorder.tensorboard(self.global_step, 'Batch/%s', suffix, **batch_kwargs)
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
        else:
            # _, token, tag, label, space, weight, disco_2d, batch_segment, segment
            b_head = [batch['tree'], batch['token']]
            if vis._draw_trees:
                weight = mean_stdev(weights).type(torch.float16).cpu().numpy()
            else:
                weight = disco_2d_logits = None

            tags   = self._model.get_decision(tag_logits  ).type(torch.short)
            labels = self._model.get_decision(label_logits).type(torch.short)
            spaces = (space - 1).type(torch.short)
            b_data = (tags, labels, spaces, weight, disco_2d_logits, stem.segment, segment)

            vis.process(batch_id, make_tensors(*b_head, *b_data))
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        # devel_bins, test_bins = self._mode_length_bins
        devel_head, test_head = self._mode_trees
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            head_trees = test_head
            if final_test:
                folder = ds_name + '_test'
                draw_trees = True
            else:
                folder = ds_name + '_test_with_devel'
                draw_trees = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
        else:
            head_trees = devel_head
            folder = ds_name + '_devel'
            draw_trees = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
        if self._optuna_mode:
            draw_trees = False
        if self.multi_corp:
            i2vs = self.i2vs[ds_name]
            m_corp = ds_name
            head_trees = head_trees[ds_name]
        else:
            i2vs = self.i2vs
            m_corp = None
        work_dir = self.recorder.create_join(folder)
        
        if serial := (draw_trees or not head_trees or self.dm is None):
            vis = DMVA(epoch, work_dir, i2vs, head_trees,
                       self.recorder.log,
                       self._evalb_lcfrs_kwargs,
                       self._discodop_prm,
                       draw_trees)
        else:
            vis = DMVP(epoch, work_dir, i2vs,
                       self._evalb_lcfrs_kwargs,
                       self._discodop_prm,
                       self.dm, m_corp)
        vis = VisRunner(vis, async_ = serial) # wrapper
        self._vis_mode = vis, use_test_set, final_test, serial

    def _get_optuna_fn(self, train_params):
        import numpy as np
        from utils.train_ops import train, get_optuna_params
        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                loss_weight = specs['train']['loss_weight']
                loss_weight['tag']   = t = trial.suggest_float('tag',   0.0, 1.0)
                loss_weight['label'] = l = trial.suggest_float('label', 0.0, 1.0)
                loss_weight['chunk'] = f = trial.suggest_float('chunk', 0.0, 1.0)
                loss_weight['disco_1d'] = d1 = trial.suggest_float('disco_1d', 0.0, 1.0)
                loss_weight['disco_2d'] = d2 = trial.suggest_float('disco_2d', 0.0, 1.0)
                loss_str = 'L=' + height_ratio(t) + height_ratio(l) + height_ratio(f) + height_ratio(d1) + height_ratio(d2)
                    
                if (d_d2a := specs['train']['disco_2d_intra_rate']) > 0:
                    specs['train']['disco_2d_intra_rate'] = d_d2a = trial.suggest_loguniform('disco_2d_intra_rate', 1e-6, d_d2a)
                    loss_weight['disco_2d_intra'] = l_d2a = trial.suggest_float('disco_2d_intra', 1e-6, 1.0)
                    loss_str += '.' + height_ratio(l_d2a)
                else:
                    loss_weight['disco_2d_intra'] = specs['train']['disco_2d_intra_rate'] = 0

                if (d_i2b := specs['train']['disco_2d_inter_rate']) > 0:
                    specs['train']['disco_2d_inter_rate'] = d_i2b = trial.suggest_loguniform('disco_2d_inter_rate', 1e-6, d_i2b)
                    loss_weight['disco_2d_inter'] = d_i2n = trial.suggest_float('disco_2d_inter', 1e-6, 1.0)
                    loss_str += ':' + height_ratio(d_i2n)
                else:
                    loss_weight['disco_2d_inter'] = specs['train']['disco_2d_inter_rate'] = 0

                medium_factor = specs['data']
                medium_factor = specs['data'][get_sole_key(medium_factor)]
                medium_factor['max_inter_height'] = mih = trial.suggest_int('max_inter_height', 0, 9)
                medium_factor = medium_factor['medium_factor']

                if involve_balanced := medium_factor['balanced'] > 0:
                    medium_factor['balanced'] = bz = trial.suggest_float('balanced', 0.0, 1.0)

                if involve_more_sub := medium_factor['more_sub'] > 0:
                    medium_factor['more_sub'] = ms = trial.suggest_float('more_sub', 0.0, 1.0)

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

                self._train_materials = (medium_factor, mih), self._train_materials[1]
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                self._train_config._nested.update(specs['train'])
                med_str = ''
                rate_str = []
                if involve_balanced: med_str += f'{height_ratio(bz)}'
                if involve_more_sub: med_str += f'{height_ratio(ms)}'
                if multi_medoid:     med_str += '[' + ''.join(height_ratio(v) for v in med_v) + ']'
                if med_str:          med_str = 'med=' + med_str
                if d_d2a > 0:       rate_str.append(f'.={d_d2a:.1e}')
                if d_i2n > 0:       rate_str.append(f':={d_i2n:.1e}')
                rate_str.append(f'lr={lr:.1e}')
                return med_str + f'h{mih};' + loss_str + ';' + ','.join(rate_str)

            self._init_mode_trees()
            self.setup_optuna_mode(spec_update_fn, trial)

            return train(optuna_params, self)['key']
        return obj_fn

from data.mp import VisRunner
from utils.file_io import remove, isdir, mkdir, listdir, join
from utils.str_ops import cat_lines, height_ratio, space_height_ratio
class DMVA(DVA):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, draw_trees):
        super().__init__(epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm)
        if draw_trees:
            draw_trees = self.join(f'tree.{epoch}.art')
            if isdir(draw_trees):
                for fname in listdir(draw_trees):
                    remove(join(draw_trees, fname))
            else:
                mkdir(draw_trees)
        self._draw_trees = draw_trees
        self._headedness_stat = self.join(f'data.{epoch}.headedness'), {}
        self._disco_2d_stat = self.join(f'data.{epoch}.2d.csv'), ['threshold,attempt,size,comp\n']

    def _process(self, batch_id, batch):
        from data.cross import draw_str_lines
        tree, token, tag, label, space, weight, disco_2d, batch_segment, segment = batch

        if self._xh_writer:
            head_lines, head_trees_for_scores = [], []
            for bt, td in tree:
                head_trees_for_scores.append(
                    inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xh_writer))
                head_lines.append(
                    '\n'.join(draw_str_lines(bt, td, label_fn = lambda i,t: t[i].label if i else f'{t[i].label} (Gold)')))
            self.save_head_trees(head_trees_for_scores, head_lines)
        else:
            head_trees_for_scores, head_lines = self.get_head_trees()
        
        # data_errors = []
        bid_offset, _ = self._evalb.total_missing
        self._evalb.add_batch_line(batch_id)
        evalb_lines = []
        for sid, (btm, td, error) in enumerate(batch_trees(token, tag, label, space, batch_segment, segment, self.i2vs, 'VROOT')):
            pred_gold = inner_score(btm, td, self._evalb_lcfrs_kwargs, self._xd_writer) + head_trees_for_scores[sid]
            if self._draw_trees: evalb_lines.append(self._evalb.add(*pred_gold))
            if error: self._v_errors[bid_offset + sid] = error
        if self._draw_trees:
            has_disco_2d = disco_2d is not None and any(l for l in disco_2d)
            c_lines = d_lines = m_lines = f'Batch #{batch_id} ───────────────────────────────────────────\n'
            c_cnt = d_cnt = m_cnt = 0
            d_has_n_comps = d_has_n_fallback = m_has_n_comps = m_has_n_fallback = 0
            _, head_stat = self._headedness_stat
            for sid, (btm, td, error, wns, stat) in enumerate(batch_trees(token, tag, label, space, batch_segment, segment, self.i2vs, 'VROOT', weight)):
                for lb, (lbc, hc) in stat.items():
                    if lb in head_stat:
                        label_cnt, head_cnts = head_stat[lb]
                        for h, c in hc.items():
                            head_cnts[h] += c
                        head_stat[lb] = lbc + label_cnt, head_cnts
                    else:
                        head_stat[lb] = lbc, hc
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
                if has_disco_2d and (s_disco_2d := [(lid, d2d[sid]) for lid, d2d in enumerate(disco_2d) if sid in d2d]):
                    if error and (eid := error[0]) > 0 and any(lid <= eid for lid, _ in s_disco_2d):
                        s_disco_2d = [(lid, s2d) for lid, s2d in s_disco_2d if lid <= eid]
                    base_lines = ['2D ']
                    base_lines += [''] * max(len(s2d[0]) for _, s2d in s_disco_2d)
                    _, csv_list = self._disco_2d_stat
                    for lid, (mat, _, _, n_comps, fallback, thresh, n_trial) in s_disco_2d:
                        new_lines = f'#{lid + 1}'
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
                        csv_list.append(f'{1 if fallback else thresh},{n_trial},{len(mat)},{n_comps}\n')
                    disco_2d_lines = '\n\n' + '\n'.join(base_lines)
                else:
                    disco_2d_lines = ''

                def weight_with_height(nid, top_down):
                    label = top_down[nid].label
                    if nid in wns:
                        label += f'#{wns[nid]}'
                    if nid == 0:
                        label += ' (Predicted)'
                    return label

                if bracket_match == g_num_brackets:
                    if tag_match == g_tag_count:
                        lines += 'Exact Match\n\n'
                    else:
                        lines += 'Exact Bracketing Match | ' + tag_line + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, label_fn = weight_with_height))
                    m_lines += lines + disco_2d_lines + '\n\n\n'
                    m_cnt += 1
                    m_has_n_fallback = max(m_has_n_fallback, has_n_fallback)
                    m_has_n_comps = max(m_has_n_comps, has_n_comps)
                else:
                    lines += f'Bracketing {p_num_brackets} > {bracket_match} < {g_num_brackets} | '
                    lines += tag_line + '\n\n'
                    lines += head_lines[sid] + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, label_fn = weight_with_height))
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

    def _after(self):
        fname, head_stat = self._headedness_stat
        with open(fname, 'w') as fw:
            for label, (label_cnt, head_cnts) in sorted(head_stat.items(), key = lambda x: x[1][0], reverse = True):
                line = f'{label}({label_cnt})'.ljust(15)
                for h, c in sorted(head_cnts.items(), key = lambda x: x[1], reverse = True):
                    line += f'{h}({c}); '
                fw.write(line[:-2] + '\n')
        fname, csv_list = self._disco_2d_stat
        with open(fname, 'w') as fw:
            fw.writelines(csv_list)
        return super()._after()

class DMVP(DVP):
    _draw_trees = False
    
    def _process(self, batch_id, batch):
        dm, _, _, corp_key = self._args
        (_, token, tag, label, space, _, _, batch_segment, segment) = batch
        dm.batch(batch_id, self._bid_offset, batch_segment, segment, token, tag, label, space, key = corp_key)
        self._bid_offset += token.shape[0]