import torch
from utils.operator import Operator
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times #, f_score
from utils.types import M_TRAIN, BaseType, frac_open_0, frac_06, frac_close, tune_epoch_type
from models.utils import PCA, fraction
from experiments.helper import WarmOptimHelper
from data.cross.evalb_lcfrs import DiscoEvalb, ExportWriter, read_param
from utils.shell_io import has_discodop, discodop_eval, byte_style
from utils.file_io import join

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

class DiscoOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, get_dm, train_config, evalb_lcfrs_prm):
        if has_discodop():
            prompt = 'Use discodop evalb (detected)'
            color = '2'
            self._discodop_prm = evalb_lcfrs_prm
        else:
            prompt = 'Use our dccp evalb, [discodop] is not installed'
            color = '3'
            if callable(get_dm): # TODO remove this
                prompt += '\n  [WARNING] \'multiprocessing_decode\' supports only discodop.'
                get_dm = None
            self._discodop_prm = None
                
        print(byte_style(prompt, color)); recorder.log(prompt)
        super().__init__(model, get_datasets, recorder, i2vs, get_dm)
        self._mode_trees = [], []
        self._train_config = train_config
        self._tune_pre_trained = False
        self._evalb_lcfrs_kwargs = read_param(evalb_lcfrs_prm)

    def _build_optimizer(self, start_epoch):
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
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
         existences, embeddings, hiddens,
         right_direc_logits, joint_logits,
         shuffled_right_direc, shuffled_joint,
         tag_logits, label_logits, segment,
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
            if tensorboard := self.recorder._writer is not None:
                self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                    Tag = fraction(tag_match,    tag_weight),
                    Label = fraction(label_match, gold_exists),
                    Right = fraction(right_match, gold_direcs),
                    Direc = fraction(direcs == gold_direcs) if direcs is not None else None,
                    Joint = fraction(joints == gold_joints))

            batch['existence'] = gold_exists
            tb_loss_kwargs = {}
            tag_loss, label_loss = self._model.get_losses(batch, None, tag_logits, top3_label_logits, label_logits)
            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            card_losses = self._model.get_stem_loss(batch, right_direc_logits, joint_logits, self._train_config.loss_weight._udirec_strength)
            if shuffled_right_direc is None:
                assert shuffled_joint is None
                shuffled_losses = None
            else:
                shuffled_losses = self._model.get_stem_loss(batch, shuffled_right_direc, shuffled_joint, self._train_config.loss_weight.sudirec_strength)
            if self._model.orient_bits == 3:
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
            
            if tensorboard:
                self.recorder.tensorboard(self.global_step, 'Loss/%s',
                    Tag = tag_loss, Label = label_loss, Joint = joint_loss, Total = total_loss,
                    **tb_loss_kwargs)
                self.recorder.tensorboard(self.global_step, 'Batch/%s',
                    SamplePerSec = batch_len / batch_time,
                    Length = batch_len,
                    Height = batch['segments'].shape[0])
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
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
        serial = save_tensors or not head_trees or self.dm is None
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
            vis = ParallelVis(epoch, work_dir, self.i2vs, self.recorder.log, self._evalb_lcfrs_kwargs, self._discodop_prm, self.dm)

        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = async_) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads, serial
        if hasattr(self._model, 'update_static_pca'):
            self._model.update_static_pca()

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, pending_heads, serial = self._vis_mode
        if serial:
            scores, desc, logg, heads = vis.after()
            if pending_heads:
                devel_head_batchess, test_head_batchess = self._mode_trees
                if use_test_set:
                    self._mode_trees = devel_head_batchess, heads
                else:
                    self._mode_trees = heads, test_head_batchess
        else:
            scores, desc, logg = vis.after()
            
        speed_outer = float(f'{count / seconds:.1f}')
        if serial:
            dmt = speed_dm_str = ''
        else:
            dmt = self.dm.duration
            speed_dm = count / dmt
            speed_dm_str = f' ◇ {speed_dm:.1f}'
            dmt = f' ◇ {dmt:.3f}'
            desc += byte_style(speed_dm_str + 'sps.', '2')

        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        logg += f' @{speed_outer}{speed_dm_str} sps. (sym:nn {rate:.2f}; {seconds:.3f}{dmt} sec.)'
        if not final_test:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores.get('TF', 0), DF = scores.get('DF', 0), 
                                      SamplePerSec = None if serial else speed_dm)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores.get('TF', 0.0) #f_score(scores.get('TF', 0.0), scores.get('DF', 0.0))
        return scores
        
    def _get_optuna_fn(self, train_params):
        import numpy as np
        from utils.types import E_ORIF5_HEAD, E_ORIF5, O_HEAD
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
                if self._model.orient_bits == 3:
                    mute += ['_direc', '_udirec_strength']
                    loss_weight['_direc'] = loss_weight['_udirec_strength'] = 0.0
                    loss_str += f'T{height_ratio(o)}'
                else:
                    if self._model.orient_bits == 2:
                        loss_weight['_direc'] = d = trial.suggest_float('_direc', 0.0, 1.0)
                        loss_weight['_udirec_strength'] = u = trial.suggest_float('_udirec_strength', 0.0, 1.0)
                        loss_str += f'D{height_ratio(o)}{height_ratio(d)}{height_ratio(u)}'
                    else:
                        mute += ['_direc', '_udirec_strength', 'shuffled__direc', 'sudirec_strength']
                        loss_str += f'S{height_ratio(o)}'

                data = specs['data']
                data = data['tiger' if 'tiger' in data else 'dptb']
                if data['ply_shuffle'] is None:
                    mute += ['shuffled_joint', 'shuffled_orient']
                else:
                    loss_weight['shuffled_joint']  = sj = trial.suggest_float('shuffled_joint',  0.0, 1.0)
                    loss_weight['shuffled_orient'] = so = trial.suggest_float('shuffled_orient', 0.0, 1.0)
                    loss_str += f'X{height_ratio(sj)}{height_ratio(so)}'

                for mute in mute:
                    loss_weight[mute] = 0
                involve_head = data['binarization']['head'] > 0
                E_ORIF = E_ORIF5_HEAD if involve_head else E_ORIF5
                binarization = np.array([trial.suggest_loguniform(x, 1e-6, 1e3) for x in E_ORIF])
                binarization /= np.sum(binarization)
                bz = {k:float(v) for k, v in zip(E_ORIF, binarization)}
                if not involve_head: bz[O_HEAD] = 0
                data['binarization'] = bz
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                self._train_config._nested.update(specs['train'])
                self._train_materials = bz, self._train_materials[1] # for train/train_initials(max_epoch>0)
                bin_str = 'bin=' + ''.join(height_ratio(x) for x in binarization)
                return bin_str + ';' + loss_str + f';lr={lr:.1e}'

            self._mode_trees = [], [] # force init
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from utils.vis import BaseVis, VisRunner
from utils.file_io import isfile
from utils.pickle_io import pickle_dump, pickle_load
from data.cross import bracketing, Counter, new_word_label, filter_words
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

def inner_score(bt, td, prm_args, export_writer = None):
    if export_writer:
        export_writer.add(bt, td)
    bt, td = new_word_label(bt, td, word_fn = prm_args.word_fn, label_fn = prm_args.label_fn)
    filter_words(bt, td, prm_args.DELETE_WORD)
    brac_cnt, brac_mul = bracketing(bt, td, excluded_labels = prm_args.DELETE_LABEL) if td else Counter()
    return brac_cnt, brac_mul, set(bt)

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
from copy import deepcopy
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
            for bt, td, error in batch_trees(bid_offset, heads, h_segment, i2vs):
                assert not error
                head_top_downs.append(deepcopy(td)) # td.copy() not works
                head_trees_for_scores.append(inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xh_writer))
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
        for sid, (bt, td, error) in enumerate(batch_trees(bid_offset, data, d_segment, i2vs, 'VROOT')):
            data_errors.append(error)
            data_top_downs.append(deepcopy(td))
            data_trees_for_scores.append(inner_score(bt, td, self._evalb_lcfrs_kwargs, self._xd_writer))
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

class ParallelVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, logger, evalb_lcfrs_kwargs, discodop_prm, dm):
        super().__init__(epoch)
        self._dtv = Dummy(work_dir, i2vs)
        self._pending_heads = False
        assert discodop_prm
        self._v_errors = {}
        self._args = dm, discodop_prm, evalb_lcfrs_kwargs, logger
        self._bid_offset = 1

    def _before(self):
        self._args[0].timeit()

    def _process(self, batch_id, batch):
        (d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc) = batch
        self._args[0].batch(batch_id, self._bid_offset, d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc)
        self._bid_offset += h_token.shape[0]

    def _after(self):
        fhead = self._dtv.join('head.export')
        fdata = self._dtv.join(f'data.{self.epoch}.export')
        dm, discodop_prm, evalb_lcfrs_kwargs, logger = self._args
        
        tree_text = dm.batched()
        if tree_text: # 'None' means 'text concat' without a memory travel
            with open(fdata, 'w') as fw:
                fw.write(tree_text)

        with open(self._dtv.join(f'eval.{self.epoch}.rpt'), 'w') as fw:
            scores = discodop_eval(fhead, fdata, discodop_prm, fw)

        scores['N'] = self._bid_offset
        tp, tr, tf, dp, dr, df = (scores[k] for k in ('TP', 'TR', 'TF', 'DP', 'DR', 'DF'))
        desc_for_screen = f'Evalb({tp:.2f}/{tr:.2f}/' + byte_style(f'{tf:.2f}', underlined = True)
        desc_for_screen += f'|{dp:.2f}/{dr:.2f}/' + byte_style(f'{df:.2f}', underlined = True) + ')'
        desc_for_logger = f'N: {self._bid_offset} Evalb({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
        return scores, desc_for_screen, desc_for_logger

    @property
    def save_tensors(self):
        return False

    @property
    def length_bins(self):
        return None