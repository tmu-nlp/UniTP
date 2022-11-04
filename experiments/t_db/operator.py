from utils.math_ops import is_bin_times #, f_score
from utils.str_ops import height_ratio
from utils.types import M_TRAIN, BaseType, frac_open_0, frac_06, frac_close, tune_epoch_type
from models.utils import PCA, fraction
from experiments.helper import make_tensors
from experiments.helper.do import DVA, DVP, DO
from data.cross.mp import b_batch_trees as batch_trees
from data.cross.binary import X_DIR, X_RGT
from data.cross.evalb_lcfrs import summary_from_add_tuples
from time import time
import torch

train_type = dict(loss_weight = dict(tag    = BaseType(0.3, validator = frac_open_0),
                                     label  = BaseType(0.1, validator = frac_open_0),
                                     joint  = BaseType(0.6, validator = frac_open_0),
                                     orient = BaseType(0.6, validator = frac_open_0),
                                     _direc = BaseType(0.6, validator = frac_open_0),
                                     _udirec_strength = BaseType(0.9, validator = frac_close),
                                     shuffled_joint   = BaseType(0.6, validator = frac_open_0),
                                     shuffled_orient  = BaseType(0.6, validator = frac_open_0),
                                     shuffled__direc  = BaseType(0.6, validator = frac_open_0),
                                     sudirec_strength = BaseType(0.9, validator = frac_close)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))

class DBOperator(DO):
    def _step(self, mode, ds_name, batch, batch_id = None):
        # assert ds_name == C_ABSTRACT
        if mode == M_TRAIN:
            gold_rights = (batch['xtype'] & X_RGT) > 0
            gold_direcs = (batch['xtype'] & X_DIR) > 0
            gold_joints = batch['joint']
            batch['supervision'] = gold_rights, gold_joints, gold_direcs
        batch['key'] = corp = ds_name if self.multi_corp else None
        # layers_of_existence, layers_of_base, layers_of_hidden, layers_of_right_direc, layers_of_joint, tags, labels, segment, seg_length
        #(right_direc, joint, shuffled_right_direc, shuffled_joint, segment, seg_length)
        batch_time = time()
        bottom, stem, tag_label = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time
        batch_size, batch_len = bottom[:2]
        (right_direc_logits, joint_logits,
         shuffled_right_direc, shuffled_joint,
         segment) = stem.extension
        _, tag_end, tag_logits, label_logits, _ = tag_label

        if mode == M_TRAIN:
            tags   = self._model.get_decision(tag_logits)
            labels = self._model.get_decision(label_logits)
            tag_existence = stem.existence[:, :tag_end]
            tag_mis      = (tags    != batch['tag'])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | tag_existence)
            label_weight = (label_mis | stem.existence)
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, stem.existence, corp, 1)
            rights, joints, direcs = self._model.stem.get_stem_prediction(right_direc_logits, joint_logits)

            batch['existence'] = stem.existence
            batch['right'] = gold_rights
            batch['direc'] = gold_direcs
            batch['joint'] = gold_joints
            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            card_losses = self._model.stem.get_stem_loss(batch, right_direc_logits, joint_logits, self._train_config.loss_weight._udirec_strength)
            if shuffled_right_direc is None:
                assert shuffled_joint is None
                shuffled_losses = None
            else:
                shuffled_losses = self._model.stem.get_stem_loss(batch, shuffled_right_direc, shuffled_joint, self._train_config.loss_weight.sudirec_strength)
            tb_loss_kwargs = {}
            if self._model.stem.orient_bits == 3:
                orient_loss, joint_loss = card_losses
                if shuffled_losses is not None:
                    shuffled_orient_loss, shuffled_joint_loss = shuffled_losses
                    tb_loss_kwargs['ShuffledOrient'] = shuffled_orient_loss
                    tb_loss_kwargs['ShuffledJoint']  = shuffled_joint_loss
                tb_loss_kwargs['Orient'] = orient_loss
            else:
                orient_loss, joint_loss, direc_loss = card_losses
                tb_loss_kwargs['Right'] = orient_loss
                if direc_loss is not None:
                    tb_loss_kwargs['Direc'] = direc_loss
                    total_loss = self._train_config.loss_weight._direc * direc_loss + total_loss
                if shuffled_losses is not None:
                    shuffled_orient_loss, shuffled_joint_loss, shuffled_direc_loss = shuffled_losses
                    tb_loss_kwargs['ShuffledRight'] = shuffled_orient_loss
                    tb_loss_kwargs['ShuffledJoint'] = shuffled_joint_loss
                    if shuffled_direc_loss is not None:
                        tb_loss_kwargs['ShuffledDirec'] = shuffled_direc_loss
                        total_loss = self._train_config.loss_weight.shuffled_direc * shuffled_direc_loss + total_loss
            if shuffled_losses is not None:
                total_loss = self._train_config.loss_weight.shuffled_joint * shuffled_joint_loss + total_loss
                total_loss = self._train_config.loss_weight.shuffled_orient * shuffled_orient_loss + total_loss
            total_loss = self._train_config.loss_weight.orient * orient_loss + total_loss
            total_loss = self._train_config.loss_weight.joint * joint_loss + total_loss
            total_loss.backward()
            
            if self.recorder._writer is not None:
                suffix = ds_name if self.multi_corp else None
                self.recorder.tensorboard(self.global_step, 'Accuracy/%s', suffix,
                    Tag   = 1 - fraction(tag_mis,     tag_weight),
                    Label = 1 - fraction(label_mis, label_weight),
                    Right = fraction((rights == gold_rights) & gold_direcs, gold_direcs),
                    Direc = fraction(direcs == gold_direcs) if direcs is not None else None,
                    Joint = fraction(joints == gold_joints))
                self.recorder.tensorboard(self.global_step, 'Loss/%s', suffix,
                    Tag = tag_loss,
                    Label = label_loss,
                    Joint = joint_loss,
                    Total = total_loss,
                    **tb_loss_kwargs)
                self.recorder.tensorboard(self.global_step, 'Batch/%s', suffix,
                    SamplePerSec = batch_len / batch_time,
                    Length = batch_len,
                    Height = batch['batch_segment'].shape[0])
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
        else:
            b_head = [batch['tree'], batch['token'], stem.segment, segment]
            if (vis := self._vis_mode[0]).save_tensors:
                if (pca := (self._model.get_static_pca(corp) if hasattr(self._model, 'get_static_pca') else None)) is None:
                    pca = PCA(stem.embedding.reshape(-1, stem.embedding.shape[2]))
                b_head += [pca(bottom.embedding).type(torch.float16), pca(stem.embedding).type(torch.float16)]
                tag_scores,     tags = self._model.get_decision_with_value(tag_logits)
                label_scores, labels = self._model.get_decision_with_value(label_logits)
                (rights, joints, direcs, right_scores, joint_scores,
                 direc_scores) = self._model.stem.get_stem_prediction(right_direc_logits, joint_logits, get_score = True)
                if direc_scores is None: direc_scores = torch.ones_like(right_scores)
                extra = [x.type(torch.float16) for x in (tag_scores, label_scores, right_scores, joint_scores, direc_scores)]
            else:
                tags   = self._model.get_decision(tag_logits  )
                labels = self._model.get_decision(label_logits)
                rights, joints, direcs = self._model.stem.get_stem_prediction(right_direc_logits, joint_logits)
                extra = None
            if direcs is None:
                direcs = torch.ones_like(rights)
            xtypes = (X_RGT * rights | direcs * X_DIR).type(torch.uint8)
            b_data = [tags.type(torch.short), labels.type(torch.short), xtypes, joints, rights, direcs]
            if extra is not None: b_data.extend(extra)
            vis.process(batch_id, make_tensors(*b_head, *b_data))
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        epoch_major, epoch_minor = epoch.split('.')
        devel_head, test_head = self._mode_trees
        if use_test_set:
            head_trees = test_head
            if final_test:
                folder = ds_name + '_test'
                save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(epoch_major)) and int(epoch_minor) == 0
        else:
            head_trees = devel_head
            folder = ds_name + '_devel'
            save_tensors = is_bin_times(int(epoch_major)) and int(epoch_minor) == 0
        if self._optuna_mode:
            save_tensors = False
        if self.multi_corp:
            head_trees = head_trees[ds_name]
            i2vs = self.i2vs[ds_name]
            m_corp = ds_name
        else:
            i2vs = self.i2vs
            m_corp = None
        if hasattr(self._model, 'update_static_pca'):
            self._model.update_static_pca(m_corp)
        work_dir = self.recorder.create_join(folder)
        if serial := (save_tensors or self.dm is None or not head_trees):
            vis = DBVA(epoch, work_dir, i2vs,
                       self.recorder.log,
                       self._evalb_lcfrs_kwargs,
                       self._discodop_prm,
                       self._model.stem.threshold,
                       save_tensors,
                       head_trees)
        else:
            vis = DBVP(epoch, work_dir, i2vs,
                       self._evalb_lcfrs_kwargs,
                       self._discodop_prm,
                       self.dm, m_corp)

        vis = VisRunner(vis, async_ = serial) # wrapper
        self._vis_mode = vis, use_test_set, final_test, serial
        
    def _get_optuna_fn(self, train_params):
        import numpy as np
        from utils.types import K_CORP, F_PHRASE, F_SENTENCE, F_CNF
        from utils.train_ops import train, get_optuna_params
        from utils.math_ops import log_to_frac

        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                loss_weight = specs['train']['loss_weight']
                loss_weight['tag']   = t = trial.suggest_float('tag',   0.0, 1.0)
                loss_weight['label'] = l = trial.suggest_float('label', 0.0, 1.0)
                loss_weight['joint'] = j = trial.suggest_float('joint', 0.0, 1.0)
                loss_weight['orient'] = o = trial.suggest_float('orient', 0.0, 1.0)
                loss_str = f'L={height_ratio(t)}{height_ratio(l)}{height_ratio(j)}'
                mute = []
                if (orient_bits := self._model.stem.orient_bits) == 3:
                    mute += '_direc', '_udirec_strength'
                    loss_weight['_direc'] = loss_weight['_udirec_strength'] = 0.0
                    loss_str += f'T{height_ratio(o)}'
                elif orient_bits == 2:
                    loss_weight['_direc'] = d = trial.suggest_float('_direc', 0.0, 1.0)
                    loss_weight['_udirec_strength'] = u = trial.suggest_float('_udirec_strength', 0.0, 1.0)
                    loss_str += f'D{height_ratio(o)}{height_ratio(d)}{height_ratio(u)}'
                else:
                    mute += '_direc', 'shuffled__direc', '_udirec_strength'
                    loss_str += f'S{height_ratio(o)}'

                data = specs['data']
                if all(dc['ply_shuffle'] is None for dc in data[K_CORP].values()):
                    mute += 'shuffled_joint', 'shuffled_orient', 'shuffled__direc', 'sudirec_strength'
                else:
                    loss_weight['shuffled_joint']  = sj = trial.suggest_float('shuffled_joint',  0.0, 1.0)
                    loss_weight['shuffled_orient'] = so = trial.suggest_float('shuffled_orient', 0.0, 1.0)
                    loss_str += f'X{height_ratio(sj)}{height_ratio(so)}'
                    if orient_bits == 2:
                        loss_weight['shuffled__direc']  = sd = trial.suggest_float('shuffled__direc',  0.0, 1.0)
                        loss_weight['sudirec_strength'] = su = trial.suggest_float('sudirec_strength', 0.0, 1.0)
                        loss_str += height_ratio(sd) + height_ratio(su)
                    else:
                        mute += 'shuffled__direc', 'sudirec_strength'

                for mute in set(mute):
                    loss_weight[mute] = 0

                new_factor = {}
                desc = []
                for corp, factor in data[K_CORP].items():
                    desc_ = [corp + '.']
                    level, left, _ = factor['binarization'].split()
                    if level == F_SENTENCE and left == F_CNF:
                        binarization = trial.suggest_float(corp + '.cnf', 0.0, 1.0)
                        factor['binarization'] = f'{level} {left} {binarization}'
                        desc_.append(height_ratio(binarization))
                        binarization = level, F_CNF, binarization
                    else:
                        if level == F_PHRASE:
                            level = trial.suggest_categorical(corp + '.l', [F_SENTENCE, F_PHRASE])
                            desc_.append(level[0])
                        beta_l = trial.suggest_float(corp + '.beta_l', 1e-2, 1e2, log = True)
                        beta_r = trial.suggest_float(corp + '.beta_r', 1e-2, 1e2, log = True)
                        factor['binarization'] = f'{level} {beta_l} {beta_r}'
                        binarization = level, beta_l, beta_r
                        desc_.append('β' + height_ratio(log_to_frac(beta_l, 1e-2, 1e2)) + height_ratio(log_to_frac(beta_r, 1e-2, 1e2)))
                    factor['esub'] = esub = trial.suggest_float(corp + '.e', 0.0, 1.0)
                    desc_.append('∅' + height_ratio(esub))
                    if (msub := factor['msub']) or isinstance(binarization[1], float):
                        factor['msub'] = msub = trial.suggest_float(corp + '.m', 0.0, 1 if msub == 0 else msub)
                        desc_.append(height_ratio(msub))
                    new_factor[corp] = binarization, esub, msub
                    desc.append(''.join(desc_))

                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_float('learning_rate', 1e-6, lr, log = True)
                self._train_config._nested.update(specs['train'])
                self._train_materials = new_factor, self._train_materials[1] # for train/train_initials(max_epoch>0)
                return ''.join(desc) + ';' + loss_str + f';lr={lr:.1e}'

            self._init_mode_trees()
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from data.mp import VisRunner
from utils.file_io import isfile, join
from utils.pickle_io import pickle_dump, pickle_load
tag_root_fn = lambda i,t: t[i].label if i else f'{t[i].label} (Predicted)'

from visualization import DiscontinuousTensorVis
class DBVA(DVA):
    def __init__(self, epoch, work_dir, i2vs, logger, evalb_lcfrs_kwargs, discodop_prm, thresholds, save_tensors, head_trees):
        super().__init__(epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, save_tensors)
        self.save_tensors = DiscontinuousTensorVis(work_dir, i2vs, thresholds) if save_tensors else None

    def _process(self, batch_id, batch):

        if self.save_tensors:
            (tree, token, batch_segment, segment,
             mpc_word, mpc_phrase, tag, label, xtype, joint, right, direc,
             tag_score, label_score, right_score, joint_score, direc_score) = batch
        else:
            (tree, token, batch_segment, segment,
             tag, label, xtype, joint, right, direc) = batch

        batch_args = token, tag, label, xtype, joint, batch_segment, segment
        hdio = self.head_data_io(batch_id, tree, batch_trees, batch_args)
        if self.pending_head and self.save_tensors:
            self.save_tensors.set_head(batch_id, token.shape[1], token, tree)

        if self.save_tensors:
            fname = self.join('summary.pkl')
            _, _, tf, _, _, df = summary_from_add_tuples(hdio.evalb_lines)
            smy = pickle_load(fname) if isfile(fname) else {}
            smy[(batch_id, self.epoch)] = dict(F1 = tf, DF = df)
            pickle_dump(fname, smy)
            trees_and_errors = tuple(zip(*hdio.trees_and_errors))
            data_trees  = trees_and_errors[:2]
            data_errors = trees_and_errors[2]
            self.save_tensors.set_data(batch_id, self.epoch, data_trees, 
                tag, label, right, joint, direc, batch_segment, segment,
                mpc_word, mpc_phrase, data_errors, hdio.evalb_lines,
                tag_score, label_score, right_score, joint_score, direc_score)
            
            fpath, draw_str_lines = self._draw_trees
            c_lines = d_lines = m_lines = f'Batch #{batch_id} ───────────────────────────────────────────\n'
            c_cnt = d_cnt = m_cnt = 0

            for sid, (btm, td, error) in enumerate(batch_trees(*batch_args, self.i2vs, 'VROOT', perserve_sub = True)):
                bracket_match, p_num_brackets, g_num_brackets, dbm, pdbc, gdbc, tag_match, g_tag_count = hdio.evalb_lines[sid]
                lines = f'Sent #{batch_id}.{sid} | #{hdio.bid_offset + sid}: '
                tag_line = 'Exact Tagging Match' if tag_match == g_tag_count else f'Tagging: {tag_match}/{g_tag_count}'
                if pdbc or gdbc:
                    tag_line += ' | DISC.'
                    if not dbm and gdbc:
                        tag_line += ' failed'
                    if not gdbc and pdbc:
                        tag_line += ' overdone'

                if bracket_match == g_num_brackets:
                    lines += f' > {bracket_match} < | '
                    if tag_match == g_tag_count:
                        lines += 'Exact Match\n\n'
                    else:
                        lines += 'Exact Bracketing Match | ' + tag_line + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, label_fn = tag_root_fn))
                    m_lines += lines + '\n\n\n'
                    m_cnt += 1
                else:
                    lines += f'Bracketing {p_num_brackets} > {bracket_match} < {g_num_brackets} | '
                    lines += tag_line + '\n\n'
                    lines += hdio.head_lines[sid] + '\n\n'
                    lines += '\n'.join(draw_str_lines(btm, td, label_fn = tag_root_fn))
                    if pdbc or gdbc:
                        d_lines += lines + '\n\n\n'
                        d_cnt += 1
                    else:
                        c_lines += lines + '\n\n\n'
                        c_cnt += 1
            fname_prefix = join(fpath, f'{batch_id:03d}.')
            total = c_cnt + d_cnt + m_cnt
            for suffix, lines, cnt in zip('cdm', (c_lines, d_lines, m_lines), (c_cnt, d_cnt, m_cnt)):
                if cnt > 0:
                    suffix = '.' + suffix + '.art'
                    with open(fname_prefix + f'{height_ratio(cnt / total)}' + suffix, 'w') as fw:
                        fw.write(lines)


class DBVP(DVP):
    save_tensors = False
    def _process(self, batch_id, batch):
        dm, _, _, corp_key = self._args
        (_, token, batch_segment, segment,
         tag, label, xtype, joint, _, _) = batch
        dm.batch(batch_id, self._bid_offset, batch_segment, segment, token, tag, label, xtype, joint, key = corp_key)
        self._bid_offset += token.shape[0]


        # if self._vd_lines:
        #     with open(self._dtv.join(f'ascii.{self.epoch}.art'), 'w') as fw:
        #         for sid, h_lines in self._vh_lines.items():
        #             fw.write(f'Key sentence #{sid}:')
        #             d_lines = self._vd_lines[sid]
        #             if d_lines is None:
        #                 fw.write(' [*** Answer Parsing Is Lacking ***]\n▚▞▚ ')
        #                 fw.write('\n▚▞▚ '.join(h_lines) + '\n\n\n\n')
        #             elif d_lines == h_lines:
        #                 fw.write(' (~ Exactly Matching Answer Parsing ~)\n███ ')
        #                 fw.write('\n███ '.join(h_lines) + '\n\n\n\n')
        #             else:
        #                 fw.write('\nK<<  ' + '\nK<<  '.join(h_lines) + '\n')
        #                 fw.write('|||\n|||\nAnswer Parsing:\nA>>  ' + '\nA>>  '.join(d_lines) + '\n\n\n\n')