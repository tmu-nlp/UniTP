import torch
from time import time
from utils.math_ops import is_bin_times #, f_score
from utils.types import M_TRAIN, BaseType, frac_open_0, frac_06, frac_close, tune_epoch_type
from models.utils import PCA, fraction
from experiments.helper import make_tensors
from experiments.helper.do import DO
from data.cross.mp import DVA, DVP, inner_score, b_batch_trees as batch_trees
from data.cross.binary import X_DIR, X_RGT
from data.cross.evalb_lcfrs import DiscoEvalb

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
            tag_weight   = stem.existence[:, :tag_end]
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, stem.existence, corp)
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
                    Tag =   fraction(tags   == batch['tag'],   tag_weight),
                    Label = fraction(labels == batch['label'], stem.existence),
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
                    Height = batch['segments'].shape[0])
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
        from utils.types import E_ORIF5_HEAD, E_ORIF5, O_HEAD, F_RAND_CON, F_RAND_CON_SUB, F_RAND_CON_MSB
        from utils.train_ops import train, get_optuna_params
        from utils.str_ops import height_ratio

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
                data = data['tiger' if 'tiger' in data else 'dptb']
                if data['ply_shuffle'] is None:
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

                binarization = data['binarization']
                if binarization[F_RAND_CON]:
                    beta_0 = trial.suggest_loguniform('beta_0', 1e-3, 1e3) # even around 1e0!
                    beta_1 = trial.suggest_loguniform('beta_1', 1e-3, 1e3)
                    bin_str = f'bin=β{beta_0:.1e},{beta_1:.1e}'
                    binarization[F_RAND_CON] = f'{beta_0}, {beta_1}'
                    if sub := binarization[F_RAND_CON_SUB]:
                        binarization[F_RAND_CON_SUB] = sub = trial.suggest_float('sub', 0.0, 1.0)
                        bin_str += height_ratio(sub)
                    if msb := binarization[F_RAND_CON_MSB]:
                        binarization[F_RAND_CON_MSB] = msb = trial.suggest_float('msb', 0.0, 1.0)
                        bin_str += height_ratio(msb)
                    bz = {F_RAND_CON: (beta_0, beta_1), F_RAND_CON_SUB: sub, F_RAND_CON_MSB: msb}
                else:
                    involve_head = binarization['head'] > 0
                    E_ORIF = E_ORIF5_HEAD if involve_head else E_ORIF5
                    binarization = np.array([trial.suggest_loguniform(x, 1e-6, 1e3) for x in E_ORIF])
                    binarization /= np.sum(binarization)
                    bin_str = 'bin=' + ''.join(height_ratio(x) for x in binarization)
                    bz = {k:float(v) for k, v in zip(E_ORIF, binarization)}
                    if not involve_head: bz[O_HEAD] = 0
                    data['binarization'] = bz
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                self._train_config._nested.update(specs['train'])
                self._train_materials = bz, self._train_materials[1] # for train/train_initials(max_epoch>0)
                return bin_str + ';' + loss_str + f';lr={lr:.1e}'

            self._init_mode_trees()
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from data.mp import VisRunner
from utils.file_io import isfile
from utils.pickle_io import pickle_dump, pickle_load

from visualization import DiscontinuousTensorVis
class DBVA(DVA):
    def __init__(self, epoch, work_dir, i2vs, logger, evalb_lcfrs_kwargs, discodop_prm, thresholds, save_tensors, head_trees):
        super().__init__(epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm)
        self.save_tensors = DiscontinuousTensorVis(work_dir, i2vs, thresholds) if save_tensors else None

    def _process(self, batch_id, batch):

        bid_offset, _ = self._evalb.total_missing
        if self.save_tensors:
            (tree, token, batch_segment, segment,
             mpc_word, mpc_phrase, tag, label, xtype, joint, right, direc,
             tag_score, label_score, right_score, joint_score, direc_score) = batch
        else:
            (tree, token, batch_segment, segment,
             tag, label, xtype, joint, right, direc) = batch

        if self._xh_writer:
            head_trees_for_scores = [inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xh_writer) for bt, td in tree]
            self.save_head_trees(head_trees_for_scores)
            if self.save_tensors:
                self.save_tensors.set_head(batch_id, token.shape[1], token, tree)
        else:
            head_trees_for_scores = self.get_head_trees()

        
        data_trees = []
        data_errors = []
        data_trees_for_scores = []
        data = zip(segment, token, tag, label, xtype, joint)
        for sid, (bt, td, error) in enumerate(batch_trees(bid_offset, data, batch_segment, self.i2vs, 'VROOT')):
            data_trees.append((td, td))
            data_errors.append(error)
            data_trees_for_scores.append(inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xd_writer))
            if error: self._v_errors[sid] = error
        scores = []
        evalb = DiscoEvalb()
        self._evalb.add_batch_line(batch_id)
        for gold, prediction in zip(head_trees_for_scores, data_trees_for_scores):
            self._evalb.add(*prediction, *gold)
            scores.append(evalb.add(*prediction, *gold))

        if self.save_tensors:
            fname = self.join('summary.pkl')
            _, _, tf, _, _, df = evalb.summary()
            smy = pickle_load(fname) if isfile(fname) else {}
            smy[(batch_id, self.epoch)] = dict(F1 = tf, DF = df)
            pickle_dump(fname, smy)
            self.save_tensors.set_data(batch_id, self.epoch, data_trees, tag, label, right, joint, direc, batch_segment, segment, mpc_word, mpc_phrase, data_errors, scores, tag_score, label_score, right_score, joint_score, direc_score)


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