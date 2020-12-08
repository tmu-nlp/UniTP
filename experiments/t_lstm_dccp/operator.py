import torch
from torch import nn
from utils.operator import Operator
from data.delta import get_rgt, get_dir, get_jnt
from utils.param_ops import get_sole_key
from time import time
from utils.str_ops import strange_to
from utils.math_ops import is_bin_times, f_score
from utils.types import M_TRAIN, BaseType, frac_open_0, frac_06, frac_close
from utils.types import str_num_array, true_type, tune_epoch_type
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper import warm_adam
from data.cross.evalb_lcfrs import DiscoEvalb, ExportWriter, read_param
from utils.shell_io import has_discodop, discodop_eval, byte_style
from utils.file_io import join, basename

train_type = dict(loss_weight = dict(tag    = BaseType(0.3, validator = frac_open_0),
                                     label  = BaseType(0.1, validator = frac_open_0),
                                     _right = BaseType(0.6, validator = frac_open_0),
                                     _direc = BaseType(0.6, validator = frac_open_0),
                                     joint  = BaseType(0.6, validator = frac_open_0),
                                     orient = BaseType(0.6, validator = frac_open_0),
                                     shuffled = BaseType(0.6, validator = frac_open_0),
                                     _undirect_orient = BaseType(0.9, validator = frac_close)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained_from_nth_epoch = tune_epoch_type,
                  lr_factor_for_tuning = frac_06)

class DiscoOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, train_config, evalb_lcfrs_prm):
        if has_discodop():
            prompt = 'Use discodop evalb (detected)'
            color = '2'
            self._discodop_prm = evalb_lcfrs_prm
        else:
            prompt = 'Use our dccp evalb, [discodop] is not installed'
            color = '3'
            self._discodop_prm = None
        print(byte_style(prompt, color)); recorder.log(prompt)
        super().__init__(model, get_datasets, recorder, i2vs)
        self._mode_trees = [], []
        self._train_config = train_config
        self._tune_pre_trained = False
        evalb_lcfrs_prm = read_param(evalb_lcfrs_prm)
        eq_w = evalb_lcfrs_prm.EQ_WORD
        eq_l = evalb_lcfrs_prm.EQ_LABEL
        self._evalb_lcfrs_kwargs = dict(unlabel = None if evalb_lcfrs_prm.LABELED else 'X',
                                        excluded_labels = evalb_lcfrs_prm.DELETE_LABEL,
                                        excluded_words  = evalb_lcfrs_prm.DELETE_WORD,
                                        equal_words     = {w:ws[-1] for ws in eq_w for w in ws},
                                        equal_labels    = {l:ls[-1] for ls in eq_l for l in ls})

    def _build_optimizer(self, start_epoch):
        # self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.6 betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6
        optim, schedule_lr = warm_adam(self._model, self._train_config.learning_rate)
        self._schedule_lr = schedule_lr
        self.recorder.init_tensorboard()
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained_from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.lr_factor_for_tuning if tune else 1
        learning_rate = self._schedule_lr(epoch, wander_ratio, lr_factor)
        self.recorder.tensorboard(self.global_step, 'Batch/%s', Learning_Rate = learning_rate, Epoch = epoch)

    def _step(self, mode, ds_name, batch, batch_id = None):

        # assert ds_name == C_ABSTRACT
        gold_rights = batch['right']
        gold_joints = batch['joint']
        gold_direcs = batch['direc']
        gold_exists = batch.pop('existence')
        if mode == M_TRAIN:
            batch['supervised_right'] = gold_rights
            batch['supervised_joint'] = gold_joints
        # layers_of_existence, layers_of_base, layers_of_hidden, layers_of_right_direc, layers_of_joint, tags, labels, segment, seg_length
        batch_time = time()
        (batch_size, batch_len, static, top3_label_logits,
         existences, embeddings, hiddens, right_direc_logits, joint_logits, shuffled_right_direc, shuffled_joint, tag_logits, label_logits, segment,
         seq_len) = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            rights, joints, direcs = self._model.get_stem_prediction(right_direc_logits, joint_logits)
            tag_weight  = gold_exists[:, :batch_len] # small endian
            tag_match   = (tags   == batch['tag']) & tag_weight
            label_match = (labels == batch['label']) & gold_exists
            right_match = (rights == gold_rights) & gold_direcs
            
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                                      Tag = fraction(tag_match,    tag_weight),
                                      Label = fraction(label_match, gold_exists),
                                      Right = fraction(right_match, gold_direcs),
                                      Direc = fraction(direcs == gold_direcs),
                                      Joint = fraction(joints == gold_joints))
            batch['existence'] = gold_exists
            shuffled = self._train_config.loss_weight.shuffled
            tb_loss_kwargs = {}
            losses = self._model.get_losses(batch, tag_logits, top3_label_logits, label_logits, right_direc_logits, joint_logits, shuffled_right_direc, shuffled_joint, self._train_config.loss_weight._undirect_orient)
            if self._model.has_fewer_losses:
                tag_loss, label_loss, orient_loss, joint_loss, shuffled_orient_loss, shuffled_joint_loss = losses
                total_loss = self._train_config.loss_weight.tag * tag_loss
                total_loss = self._train_config.loss_weight.label * label_loss + total_loss
                total_loss = self._train_config.loss_weight.joint * joint_loss + total_loss
                total_loss = self._train_config.loss_weight.orient * orient_loss + total_loss
                if shuffled_joint_loss is not None:
                    total_loss = self._train_config.loss_weight.joint * shuffled_joint_loss * shuffled + total_loss
                    total_loss = self._train_config.loss_weight.orient * shuffled_orient_loss * shuffled + total_loss
                    tb_loss_kwargs['ShuffledOrient'] = shuffled_orient_loss
                    tb_loss_kwargs['ShuffledJoint']  = shuffled_joint_loss
                tb_loss_kwargs['Orient'] = orient_loss
            else:
                tag_loss, label_loss, right_loss, joint_loss, direc_loss, shuffled_right_loss, shuffled_joint_loss, shuffled_direc_loss = losses
                total_loss = self._train_config.loss_weight.tag * tag_loss
                total_loss = self._train_config.loss_weight.label * label_loss + total_loss
                total_loss = self._train_config.loss_weight.joint * joint_loss + total_loss
                total_loss = self._train_config.loss_weight._right * right_loss + total_loss
                total_loss = self._train_config.loss_weight._direc * direc_loss + total_loss
                if shuffled_joint_loss is not None:
                    total_loss = self._train_config.loss_weight.joint * shuffled_joint_loss * shuffled + total_loss
                    total_loss = self._train_config.loss_weight._right * shuffled_right_loss * shuffled + total_loss
                    total_loss = self._train_config.loss_weight._direc * shuffled_direc_loss * shuffled + total_loss
                    tb_loss_kwargs['ShuffledRight'] = shuffled_right_loss
                    tb_loss_kwargs['ShuffledDirec'] = shuffled_direc_loss
                    tb_loss_kwargs['ShuffledJoint'] = shuffled_joint_loss
                tb_loss_kwargs['Right'] = right_loss
                tb_loss_kwargs['Direc'] = direc_loss
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag = tag_loss, Label = label_loss, Joint = joint_loss, Total = total_loss,
                                       **tb_loss_kwargs)
            self.recorder.tensorboard(self.global_step, 'Batch/%s',
                                      SamplePerSec = batch_len / batch_time,
                                      Length = batch_len,
                                      Height = batch['segments'].shape[0])
        else:
            vis, _, _, pending_heads = self._vis_mode
            if vis.save_tensors:
                if self._model._input_layer.has_static_pca:
                    mpc_token = self._model._input_layer.pca(static)
                    mpc_label = self._model._input_layer.pca(embeddings)
                else:
                    pca = PCA(embeddings.reshape(-1, embeddings.shape[2]))
                    mpc_token = pca(static)
                    mpc_label = pca(embeddings)

                tag_scores,     tags = self._model.get_decision_with_value(tag_logits)
                label_scores, labels = self._model.get_decision_with_value(label_logits)
                rights, joints, direcs, right_scores, joint_scores, direc_scores = self._model.get_stem_prediction(right_direc_logits, joint_logits, get_score = True)
                extra = mpc_token, mpc_label, tag_scores, label_scores, right_scores, joint_scores, direc_scores
            else:
                tags    = self._model.get_decision(tag_logits  )
                labels  = self._model.get_decision(label_logits)
                rights, joints, direcs = self._model.get_stem_prediction(right_direc_logits, joint_logits)
                extra = None
            if pending_heads:
                b_head = tuple(batch[x] for x in 'segments seq_len token tag label right joint direc'.split())
                b_data = (segment, seq_len, tags, labels, rights, joints, direcs)
                tensors = b_head + b_data
            else:
                tensors = (segment, seq_len, batch['token'], tags, labels, rights, joints, direcs)
            if extra:
                tensors = extra + tensors

            tensors = tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensors)
            vis.process(batch_id, tensors)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        # devel_bins, test_bins = self._mode_length_bins
        devel_head_batchess, test_head_batchess = self._mode_trees
        if use_test_set:
            head_trees = test_head_batchess
            if final_test:
                folder = ds_name + '_test'
                save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(float(epoch)) - 1)
        else:
            head_trees = devel_head_batchess
            folder = ds_name + '_devel'
            save_tensors = is_bin_times(int(float(epoch)) - 1)
        if self._optuna_mode:
            save_tensors = False
        vis = DiscoVis(epoch,
                       self.recorder.create_join(folder),
                       self.i2vs,
                       head_trees,
                       self.recorder.log,
                       self._evalb_lcfrs_kwargs,
                       self._discodop_prm,
                       self._model.threshold,
                       save_tensors)
        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = True) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads
        if self._model._input_layer.has_static_pca:
            self._model._input_layer.flush_pc_if_emb_is_tuned()

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, pending_heads = self._vis_mode
        scores, desc, logg, heads = vis.after()
        if pending_heads:
            devel_head_batchess, test_head_batchess = self._mode_trees
            if use_test_set:
                self._mode_trees = devel_head_batchess, heads
            else:
                self._mode_trees = heads, test_head_batchess
        speed = float(f'{count / seconds:.1f}')
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        logg += f' @{speed}sps. (sym:nn {rate:.2f})'
        scores['speed'] = speed
        if not final_test:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores['TF'], Disc_F1 = scores['DF'], SamplePerSec = speed)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        if scores['TF'] + scores['DF'] > 0:
            scores['key'] = f_score(scores['TF'], scores['DF'], 0.5)
        else:
            scores['key'] = 0
        return scores
        
    def optuna_model(self, train_params):
        import optuna
        from optuna.trial import TrialState
        import numpy as np
        from utils.types import E_ORIF5_HEAD
        from utils.train_ops import train

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                data = specs['data']
                data = data['tiger' if 'tiger' in data else 'dptb']
                loss_weight = specs['train']['loss_weight']
                loss_weight['tag']   = t = trial.suggest_float('tag',   0.0, 1.0)
                loss_weight['label'] = l = trial.suggest_float('label', 0.1, 1.0)
                loss_weight['joint'] = j = trial.suggest_float('joint', 0.1, 1.0)
                if self._model.has_fewer_losses:
                    loss_weight['orient'] = o = trial.suggest_float('orient', 0.1, 1.0)
                    loss_str = f'L={t:.1f},{l:.1f},{j:.1f};{o:.1f};'
                else:
                    loss_weight['_right'] = r = trial.suggest_float('_right', 0.1, 1.0)
                    loss_weight['_direc'] = d = trial.suggest_float('_direc', 0.1, 1.0)
                    loss_weight['_undirect_orient'] = u = trial.suggest_float('_undirect_orient', 0.1, 1.0)
                    loss_str = f'L={t:.1f},{l:.1f},{j:.1f},{r:.1f},{d:.1f},{u:.1f};'
                if data['shuffle_swap'] is not None:
                    loss_weight['shuffled'] = s = trial.suggest_float('shuffled', 0.0, 1.0)
                    loss_str += f'{s:.1f}'
                binarization = np.array([trial.suggest_loguniform(x, 1e-5, 1e5) for x in E_ORIF5_HEAD])
                binarization /= np.sum(binarization)
                data['binarization'] = bz = {k:float(v) for k, v in zip(E_ORIF5_HEAD, binarization)}
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
                self._train_config._nested.update(specs['train'])
                self._train_materials = bz, self._train_materials[1] # for train/train_initials(max_epoch>0)
                bin_str = 'bin=' + ','.join(f'{x:.2f}' for x in binarization)
                return bin_str + ';' + loss_str + f';lr={lr:.1e}'

            self._mode_trees = [], [] # force init
            base_devel_score = self.setup_optuna_mode(spec_update_fn, trial)

            optuna_params = {} # change train_params
            optuna_params['fine_validation_at_nth_wander'] = 1
            optuna_params['stop_at_nth_wander'] = train_params.stop_at_nth_wander
            optuna_params['fine_validation_each_nth_epoch'] = 5

            optuna_params['max_epoch'] = 100
            optuna_params['update_every_n_batch'] = train_params.update_every_n_batch
            optuna_params['test_with_validation'] = False
            optuna_params['optuna_trials'] = 0
            train(optuna_params, self)
            return self._recorder.key_score # child score

        fpath = join(self._recorder.create_join(), 'trials.db')
        study = optuna.create_study(direction  = 'maximize',
                                    study_name = 'dccp-hyper',
                                    storage    = 'sqlite:///' + fpath,
                                    load_if_exists = True)
        n_trials = train_params.optuna_trials
        for tid in reversed(range(len(study.trials))):
            t = study.trials[tid]
            if t.state in (TrialState.PRUNED, TrialState.COMPLETE):
                n_trials -= 1
            else:
                study.trials.pop(tid)
        if n_trials > 0:
            self._recorder.log(f'{n_trials} trials left to explore ...')
            study.optimize(obj_fn, n_trials = n_trials) # core
            self.restore_recorder()


from utils.vis import BaseVis, VisRunner
from utils.file_io import isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump, pickle_load
from data.cross import disco_tree, bracketing, Counter, draw_str_lines
def batch_trees(bid_offset, heads_gen, segments, i2vs, fall_back_root_label = None,
                export_writer = None,
                v_errors = None,
                unlabel = None,
                equal_labels = None,
                equal_words = None,
                excluded_words = None,
                excluded_labels = None):
    trees = []
    errors = []
    top_downs = []
    for sid, (s_seq_len, s_token, s_tag, s_label, s_right, s_joint, s_direc) in enumerate(heads_gen):
        sid += bid_offset
        layers_of_label = []
        layers_of_right = []
        layers_of_joint = []
        layers_of_direc = []
        jnt_start = 0
        rgt_start = 0
        for s_size, s_len in zip(segments, s_seq_len):
            label_layer = tuple(i2vs.label[i] for i in s_label[rgt_start + 1: rgt_start + s_len + 1])
            layers_of_label.append(label_layer)
            layers_of_joint.append(s_joint[jnt_start + 1: jnt_start + s_len])
            layers_of_right.append(s_right[rgt_start + 1: rgt_start + s_len + 1])
            layers_of_direc.append(s_direc[rgt_start + 1: rgt_start + s_len + 1])
            rgt_start += s_size
            jnt_start += s_size - 1
            if s_len == 1:
                break
        bottom_end = s_seq_len[0] + 1
        tags  = tuple(i2vs.tag[i]   for i in   s_tag[1:bottom_end])
        words = tuple(i2vs.token[i] for i in s_token[1:bottom_end])

        bottom, td, rt, error = disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, fall_back_root_label)
        if fall_back_root_label is None: # head conversion
            assert error is None
        else: # data conversion
            errors.append(error)
            if error:
                v_errors[sid] = error
        # except Exception as err:
        #     import traceback
        #     traceback.print_exc()
        #     print(err)
        #     import pdb; pdb.set_trace()
        #     bottom, td, rt = disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, fall_back_root_label)
        #     brackets_cnt = None
        #     bottom = ((bid, wd, tg) for bid, (wd, tg) in enumerate(zip(words, tags)))
        # else:
        if td:
            brackets_cnt = bracketing(bottom, td, rt, False, unlabel, excluded_labels, equal_labels)
        else:
            brackets_cnt = Counter()
        top_downs.append(td)
        bottom_set = set()
        for bid, word, tag in bottom:
            if equal_words:
                word = equal_words.get(word, word)
            if word in excluded_words:
                continue
            bottom_set.add((bid, word, tag))
        trees.append((brackets_cnt, bottom_set))
        if export_writer:
            export_writer.add(bottom, td, rt)
    if fall_back_root_label is None: # head
        return trees, top_downs
    return trees, top_downs, errors # data

from visualization import DiscontinuousTensorVis
from data.cross import explain_error
class DiscoVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, thresholds, save_tensors):
        super().__init__(epoch)
        self._evalb = DiscoEvalb()
        self._dtv = DiscontinuousTensorVis(work_dir, i2vs, thresholds)
        self._logger = logger
        self._head_batches = head_trees
        self._pending_heads = ph = not head_trees
        self._data_batch_cnt = 0
        self._evalb_lcfrs_kwargs = evalb_lcfrs_kwargs
        self._discodop_prm = discodop_prm
        self._xh_writer = ExportWriter() if ph and discodop_prm else None
        self._xd_writer = ExportWriter() if discodop_prm else None
        self._v_errors = {}
        self.register_property('save_tensors', save_tensors)

    def _before(self):
        pass

    def _process(self, batch_id, batch):

        bid_offset, _ = self._evalb.total_missing

        i2vs = self._dtv.vocabs
        if self.save_tensors:
            mpc_word, mpc_phrase, tag_score, label_score, right_score, joint_score, direc_score = batch[:7]
            batch = batch[7:]
        if self._pending_heads:
            (h_segment, h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc,
             d_segment, d_seq_len,          d_tag, d_label, d_right, d_joint, d_direc) = batch
            heads = zip(h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc)
            heads, trees = batch_trees(bid_offset, heads, h_segment, i2vs, export_writer = self._xh_writer, **self._evalb_lcfrs_kwargs)
            self._head_batches.append(heads)
            if self.save_tensors:
                self._dtv.set_head(batch_id, h_token.shape[1], h_token, h_tag, h_label, h_right, h_joint, h_direc, trees, h_segment, h_seq_len)
        else:
            (d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc) = batch
            heads = self._head_batches[self._data_batch_cnt]
            self._data_batch_cnt += 1
        
        evalb = DiscoEvalb()
        data = zip(d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc)
        data, trees, errors = batch_trees(bid_offset, data, d_segment, i2vs, 'VROOT', v_errors = self._v_errors, export_writer = self._xd_writer, **self._evalb_lcfrs_kwargs)
        scores = []
        self._evalb.add_batch_line(batch_id)
        for gold, prediction in zip(heads, data):
            scores.append(self._evalb.add(*prediction, *gold))
            evalb.add(*prediction, *gold)

        if self.save_tensors:
            fname = self._dtv.join('summary.pkl')
            _, _, tf, _, _, df = evalb.summary()
            smy = pickle_load(fname) if isfile(fname) else {}
            smy[(batch_id, self.epoch)] = dict(F1 = tf, DF = df)
            pickle_dump(fname, smy)

        if self.save_tensors:
            self._dtv.set_data(batch_id, self.epoch, h_token, d_tag, d_label, d_right, d_joint, d_direc, trees, d_segment, d_seq_len, mpc_word, mpc_phrase, errors, scores, tag_score, label_score, right_score, joint_score, direc_score)

    def _after(self):
        total_sents, num_errors = self._evalb.total_missing
        if num_errors:
            self._logger(f'  {num_errors} system errors from evalb (this should not appear in log)')
        num_errors = len(self._v_errors)
        if num_errors:
            fname = f'data.{self.epoch}.errors'
            self._logger(f'  {num_errors} system errors, check {fname} for details.')
            with open(self._dtv.join(fname), 'w') as fw:
                for sid, error_args in self._v_errors.items():
                    fw.write(explain_error(*error_args) + '\n')

        if self._xh_writer:
            self._xh_writer.dump(self._dtv.join('head.export'))
        
        with open(self._dtv.join(f'eval.{self.epoch}.rpt'), 'w') as fw:
            fw.write(str(self._evalb))

            if self._xd_writer:
                fhead = self._dtv.join('head.export')
                fdata = self._dtv.join(f'data.{self.epoch}.export')
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
        return scores, desc_for_screen, desc_for_logger, self._head_batches

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