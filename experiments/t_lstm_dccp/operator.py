import torch
from torch import nn
from utils.operator import Operator
from data.delta import get_rgt, get_dir, get_jnt
from utils.param_ops import get_sole_key
from time import time
from utils.str_ops import strange_to
from utils.math_ops import is_bin_times, f_score
from utils.types import M_TRAIN, BaseType, frac_open_0, frac_06, frac_close
from utils.types import str_num_array, true_type, false_type, tune_epoch_type
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper import WarmOptimHelper
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
                  multiprocessing_decode = true_type,
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))

class DiscoOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, train_config, evalb_lcfrs_prm):
        if has_discodop():
            prompt = 'Use discodop evalb (detected)'
            if train_config.multiprocessing_decode:
                prompt += ' +> with multiprocessing_decode'
            color = '2'
            self._discodop_prm = evalb_lcfrs_prm
        else:
            prompt = 'Use our dccp evalb, [discodop] is not installed'
            color = '3'
            self._discodop_prm = None
            if train_config.multiprocessing_decode:
                prompt += '\n  [disabled] \'train::multiprocessing_decode = true\' is not supported.'
                train_config._nested['multiprocessing_decode'] = False
        print(byte_style(prompt, color)); recorder.log(prompt)
        super().__init__(model, get_datasets, recorder, i2vs)
        self._dm = None
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
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
        self.recorder.init_tensorboard()
        optim = hp.optimizer
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained.from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.tune_pre_trained.lr_factor if tune else 1
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
                                      Direc = fraction(direcs == gold_direcs) if direcs is not None else None,
                                      Joint = fraction(joints == gold_joints))

            # if self._train_config.label_freq_as_loss_weight:
            #     label_mask = self._train_config.label_log_freq_inv[batch['label']]
            # else:
            #     label_mask = None

            batch['existence'] = gold_exists
            shuffled = self._train_config.loss_weight.shuffled
            tb_loss_kwargs = {}
            losses = self._model.get_losses(batch, None, tag_logits, top3_label_logits, label_logits, right_direc_logits, joint_logits, shuffled_right_direc, shuffled_joint, self._train_config.loss_weight._undirect_orient)
            if self._model.orient_bits == 3:
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
                if shuffled_joint_loss is not None:
                    total_loss = self._train_config.loss_weight.joint * shuffled_joint_loss * shuffled + total_loss
                    total_loss = self._train_config.loss_weight._right * shuffled_right_loss * shuffled + total_loss
                    tb_loss_kwargs['ShuffledRight'] = shuffled_right_loss
                    tb_loss_kwargs['ShuffledJoint'] = shuffled_joint_loss
                    if shuffled_direc_loss is not None:
                        tb_loss_kwargs['ShuffledDirec'] = shuffled_direc_loss
                        total_loss = self._train_config.loss_weight._direc * shuffled_direc_loss * shuffled + total_loss
                tb_loss_kwargs['Right'] = right_loss
                if direc_loss is not None:
                    tb_loss_kwargs['Direc'] = direc_loss
                    total_loss = self._train_config.loss_weight._direc * direc_loss + total_loss
            total_loss.backward()
            
            if hasattr(self._model, 'tensorboard'):
                self._model.tensorboard(self.recorder, self.global_step)
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag = tag_loss, Label = label_loss, Joint = joint_loss, Total = total_loss,
                                       **tb_loss_kwargs)
            self.recorder.tensorboard(self.global_step, 'Batch/%s',
                                      SamplePerSec = batch_len / batch_time,
                                      Length = batch_len,
                                      Height = batch['segments'].shape[0])
        else:
            vis, _, _, pending_heads, _ = self._vis_mode
            if vis.save_tensors:
                pca = self._model.get_static_pca() if hasattr(self._model, 'get_static_pca') else None
                if pca is None:
                    pca = PCA(embeddings.reshape(-1, embeddings.shape[2]))
                mpc_token = pca(static)
                mpc_label = pca(embeddings)

                tag_scores,     tags = self._model.get_decision_with_value(tag_logits)
                label_scores, labels = self._model.get_decision_with_value(label_logits)
                rights, joints, direcs, right_scores, joint_scores, direc_scores = self._model.get_stem_prediction(right_direc_logits, joint_logits, get_score = True)
                if direc_scores is None: direc_scores = torch.ones_like(right_scores)
                extra = mpc_token, mpc_label, tag_scores, label_scores, right_scores, joint_scores, direc_scores
            else:
                tags    = self._model.get_decision(tag_logits  )
                labels  = self._model.get_decision(label_logits)
                rights, joints, direcs = self._model.get_stem_prediction(right_direc_logits, joint_logits)
                extra = None
            if direcs is None:
                direcs = torch.ones_like(rights)
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
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            head_trees = test_head_batchess
            if final_test:
                folder = ds_name + '_test'
                save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(epoch_major)) and int(epoch_minor) == 0
        else:
            head_trees = devel_head_batchess
            folder = ds_name + '_devel'
            save_tensors = is_bin_times(int(epoch_major)) and int(epoch_minor) == 0
        if self._optuna_mode:
            save_tensors = False
        serial = save_tensors or not head_trees or not self._train_config.multiprocessing_decode
        work_dir = self.recorder.create_join(folder)
        if serial:
            async_ = True
            vis = DiscoVis(epoch,
                           work_dir,
                           self.i2vs,
                           head_trees,
                           self.recorder.log,
                           self._evalb_lcfrs_kwargs,
                           self._discodop_prm,
                           self._model.threshold,
                           save_tensors)
        else:
            async_ = False
            vis = ParallelVis(epoch, work_dir, self.i2vs, self.recorder.log, self._evalb_lcfrs_kwargs, self._discodop_prm, self._dm)

        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = async_) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads, serial
        if hasattr(self._model, 'update_static_pca'):
            self._model.update_static_pca()

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, pending_heads, serial = self._vis_mode
        scores, desc, logg, heads_or_dm = vis.after()
        if not serial:
            self._dm = heads_or_dm
        elif pending_heads:
            devel_head_batchess, test_head_batchess = self._mode_trees
            if use_test_set:
                self._mode_trees = devel_head_batchess, heads_or_dm
            else:
                self._mode_trees = heads_or_dm, test_head_batchess

        speed_outer = float(f'{count / seconds:.1f}')
        if serial:
            dmt = speed_dm_str = ''
        else:
            dmt = self._dm.duration
            speed_dm = count / dmt
            speed_dm_str = f' ◇ {speed_dm:.1f}'
            dmt = f' ◇ {dmt:.3f}'
            desc += byte_style(speed_dm_str + 'sps.', '2')

        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        logg += f' @{speed_outer}{speed_dm_str} sps. (sym:nn {rate:.2f}; {seconds:.3f}{dmt} sec.)'
        if final_test:
            if self._dm:
                self._dm.close()
        else:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores.get('F1', 0), SamplePerSec = None if serial else speed_dm)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores.get('TF', 0.0)
        return scores
        
    def _get_optuna_fn(self, train_params):
        import numpy as np
        from utils.types import E_ORIF5_HEAD
        from utils.train_ops import train, get_optuna_params
        from utils.str_ops import height_ratio

        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                loss_weight = specs['train']['loss_weight']
                loss_weight['tag']   = t = trial.suggest_float('tag',   0.0, 1.0)
                loss_weight['label'] = l = trial.suggest_float('label', 0.1, 1.0)
                loss_weight['joint'] = j = trial.suggest_float('joint', 0.1, 1.0)
                loss_str = f'L={height_ratio(t)}{height_ratio(l)}{height_ratio(j)}'
                if self._model.orient_bits == 3:
                    loss_weight['orient'] = o = trial.suggest_float('orient', 0.1, 1.0)
                    loss_str += f'T{height_ratio(o)}'
                else:
                    loss_weight['_right'] = r = trial.suggest_float('_right', 0.1, 1.0)
                    if self._model.orient_bits == 2:
                        loss_weight['_direc'] = d = trial.suggest_float('_direc', 0.1, 1.0)
                        loss_weight['_undirect_orient'] = u = trial.suggest_float('_undirect_orient', 0.1, 1.0)
                        loss_str += f'D{height_ratio(r)}{height_ratio(d)}{height_ratio(u)}'
                    else:
                        loss_str += f'S{height_ratio(r)}'

                data = specs['data']
                data = data['tiger' if 'tiger' in data else 'dptb']
                if data['shuffle_swap'] is not None:
                    loss_weight['shuffled'] = s = trial.suggest_float('shuffled', 0.0, 1.0)
                    loss_str += f'X{height_ratio(s)}'
                
                binarization = np.array([trial.suggest_loguniform(x, 1e-5, 1e5) for x in E_ORIF5_HEAD])
                binarization /= np.sum(binarization)
                data['binarization'] = bz = {k:float(v) for k, v in zip(E_ORIF5_HEAD, binarization)}
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
                self._train_config._nested.update(specs['train'])
                self._train_materials = bz, self._train_materials[1] # for train/train_initials(max_epoch>0)
                bin_str = 'bin=' + ''.join(height_ratio(x) for x in binarization)
                return bin_str + ';' + loss_str + f';lr={lr:.1e}'

            self._mode_trees = [], [] # force init
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from utils.vis import BaseVis, VisRunner
from utils.file_io import isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump, pickle_load
from data.cross import bracketing, Counter, draw_str_lines
from data.cross.binary import disco_tree
def batch_trees(bid_offset, heads_gen, segments, i2vs, fall_back_root_label = None):
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
        yield disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, fall_back_root_label)

def inner_score(bt, td, rt,
                export_writer = None,
                unlabel = None,
                equal_labels = None,
                equal_words = None,
                excluded_words = None,
                excluded_labels = None):
    if export_writer:
        export_writer.add(bt, td, rt)
    brackets_cnt = bracketing(bt, td, rt, False, unlabel, excluded_labels, equal_labels) if td else Counter()
    bottom_set = set()
    for bid, word, tag in bt:
        if equal_words:
            word = equal_words.get(word, word)
        if word in excluded_words:
            continue
        bottom_set.add((bid, word, tag))
    return brackets_cnt, bottom_set

class Dummy:
    def __init__(self, work_dir, i2vs):
        self._work_dir = work_dir
        self._i2vs = i2vs

    @property
    def vocabs(self):
        return self._i2vs

    def join(self, *fpath):
        return join(self._work_dir, *fpath)

from visualization import DiscontinuousTensorVis
from data.cross import explain_error
class DiscoVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs, discodop_prm, thresholds, save_tensors):
        super().__init__(epoch)
        self._evalb = DiscoEvalb()
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
        if save_tensors:
            self._dtv = DiscontinuousTensorVis(work_dir, i2vs, thresholds)
        else:
            self._dtv = Dummy(work_dir, i2vs)

    def _before(self):
        pass

    def _process(self, batch_id, batch):

        i2vs = self._dtv.vocabs
        bid_offset, _ = self._evalb.total_missing
        if self.save_tensors:
            mpc_word, mpc_phrase, tag_score, label_score, right_score, joint_score, direc_score = batch[:7]
            batch = batch[7:]

        if self._pending_heads:
            (h_segment, h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc,
             d_segment, d_seq_len,          d_tag, d_label, d_right, d_joint, d_direc) = batch
            head_top_downs = []
            head_trees_for_scores = []
            heads = zip(h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc)
            for bt, td, rt, error in batch_trees(bid_offset, heads, h_segment, i2vs):
                assert not error
                head_top_downs.append(td)
                head_trees_for_scores.append(inner_score(bt, td, rt, self._xh_writer, **self._evalb_lcfrs_kwargs))
            self._head_batches.append(head_trees_for_scores)
            if self.save_tensors:
                self._dtv.set_head(batch_id, h_token.shape[1], h_token, h_tag, h_label, h_right, h_joint, h_direc, head_top_downs, h_segment, h_seq_len)
        else:
            (d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc) = batch
            head_trees_for_scores = self._head_batches[self._data_batch_cnt]
            self._data_batch_cnt += 1
        
        data_errors = []
        data_top_downs = []
        data_trees_for_scores = []
        data = zip(d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc)
        for sid, (bt, td, rt, error) in enumerate(batch_trees(bid_offset, data, d_segment, i2vs, 'VROOT')):
            data_errors.append(error)
            data_top_downs.append(td)
            data_trees_for_scores.append(inner_score(bt, td, rt, self._xd_writer, **self._evalb_lcfrs_kwargs))
            if error: self._v_errors[sid] = error
        scores = []
        evalb = DiscoEvalb()
        self._evalb.add_batch_line(batch_id)
        for gold, prediction in zip(head_trees_for_scores, data_trees_for_scores):
            self._evalb.add(*prediction, *gold)
            scores.append(evalb.add(*prediction, *gold))

        if self.save_tensors:
            fname = self._dtv.join('summary.pkl')
            _, _, tf, _, _, df = evalb.summary()
            smy = pickle_load(fname) if isfile(fname) else {}
            smy[(batch_id, self.epoch)] = dict(F1 = tf, DF = df)
            pickle_dump(fname, smy)
            self._dtv.set_data(batch_id, self.epoch, h_token, d_tag, d_label, d_right, d_joint, d_direc, data_top_downs, d_segment, d_seq_len, mpc_word, mpc_phrase, data_errors, scores, tag_score, label_score, right_score, joint_score, direc_score)

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


from data.cross.binary import BxDM
from utils.types import num_threads
class ParallelVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger, evalb_lcfrs_kwargs, discodop_prm, dm):
        super().__init__(epoch)
        self._dtv = Dummy(work_dir, i2vs)
        self._logger = logger
        self._pending_heads = False
        self._evalb_lcfrs_kwargs = evalb_lcfrs_kwargs
        assert discodop_prm
        self._discodop_prm = discodop_prm
        self._v_errors = {}
        self._dm = dm
        self._bid_offset = 1

    def _before(self):
        if self._dm:
            self._dm.timeit()

    def _process(self, batch_id, batch):
        (d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc) = batch
        batch_size = h_token.shape[0]
        
        if self._dm is None:
            self._dm = BxDM(batch_size, self._dtv.vocabs, num_threads)
        self._dm.batch(self._bid_offset, d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc)
        self._bid_offset += batch_size

    def _after(self):
        fhead = self._dtv.join('head.export')
        fdata = self._dtv.join(f'data.{self.epoch}.export')
        
        dm = self._dm
        tree_text = dm.batched()
        if tree_text: # 'None' means 'text concat' without a memory travel
            with open(fdata, 'w') as fw:
                fw.write(tree_text)

        with open(self._dtv.join(f'eval.{self.epoch}.rpt'), 'w') as fw:
            scores = discodop_eval(fhead, fdata, self._discodop_prm, fw)

        scores['N'] = self._bid_offset
        tp, tr, tf, dp, dr, df = (scores[k] for k in ('TP', 'TR', 'TF', 'DP', 'DR', 'DF'))
        desc_for_screen = f'Evalb({tp:.2f}/{tr:.2f}/' + byte_style(f'{tf:.2f}', underlined = True)
        desc_for_screen += f'|{dp:.2f}/{dr:.2f}/' + byte_style(f'{df:.2f}', underlined = True) + ')'
        desc_for_logger = f'N: {self._bid_offset} Evalb({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
        return scores, desc_for_screen, desc_for_logger, dm

    @property
    def save_tensors(self):
        return False

    @property
    def length_bins(self):
        return None