import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.operator import Operator
from data.delta import get_rgt, get_dir, get_jnt
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06, frac_close
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper import warm_adam

train_type = dict(loss_weight = dict(tag    = BaseType(0.3, validator = frac_open_0),
                                     label  = BaseType(0.1, validator = frac_open_0),
                                     orient = BaseType(0.6, validator = frac_open_0),
                                     direct = BaseType(0.6, validator = frac_open_0),
                                     direct_for_orient = BaseType(0.9, validator = frac_close),
                                     joint  = BaseType(0.6, validator = frac_open_0)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained_from_nth_epoch = tune_epoch_type,
                  lr_factor_for_tuning = frac_06,
                  orient_hinge_loss = true_type)

class DiscoOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, train_config, evalb_lcfrs_prm):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._sigmoid = nn.Sigmoid()
        self._mode_trees = [], []
        self._train_config = train_config
        self._tune_pre_trained = False
        eq_w = evalb_lcfrs_prm.EQ_WORD
        eq_l = evalb_lcfrs_prm.EQ_LABEL
        self._evalb_lcfrs_kwargs = dict(unlabel = None if evalb_lcfrs_prm.LABELED else 'X',
                                        excluded_labels = evalb_lcfrs_prm.DELETE_LABEL,
                                        excluded_words  = evalb_lcfrs_prm.DELETE_WORD,
                                        equal_words     = {w:ws[-1] for ws in eq_w for w in ws},
                                        equal_labels    = {l:ls[-1] for ls in eq_l for l in ls})

    def _build_optimizer(self, start_epoch):
        # self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.6 betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6
        self._writer = SummaryWriter(self.recorder.create_join('train'))
        optim, schedule_lr = warm_adam(self._model, self._train_config.learning_rate)
        self._schedule_lr = schedule_lr
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained_from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.lr_factor_for_tuning if tune else 1
        learning_rate = self._schedule_lr(epoch, wander_ratio, lr_factor)
        self._writer.add_scalar('Batch/Learning_Rate', learning_rate, self.global_step)
        self._writer.add_scalar('Batch/Epoch', epoch, self.global_step)

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
         existences, embeddings, hiddens, right_direc_logits, joint_logits, tag_logits, label_logits, segment,
         seq_len) = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time

        tags    = self._model.get_decision(tag_logits  )
        labels  = self._model.get_decision(label_logits)
        right_logits = right_direc_logits[:, :, 0]
        direc_logits = right_direc_logits[:, :, 1]
        if self._train_config.orient_hinge_loss:
            rights = right_logits > 0
            direcs = direc_logits > 0
            joints = joint_logits > 0
        else:
            right_logits = self._sigmoid(right_logits)
            direc_logits = self._sigmoid(direc_logits)
            joint_logits = self._sigmoid(joint_logits)
            rights = right_logits > 0.5
            direcs = direc_logits > 0.5
            joints = joint_logits > 0.5

        if mode == M_TRAIN:
            # existence & height -> right, direc
            # all joint
            tag_mis       = (tags    != batch['tag'])
            label_mis     = (labels  != batch['label'])
            tag_weight    = (  tag_mis | gold_exists[:, :batch_len]) # small endian
            label_weight  = (label_mis | gold_exists)
            orient_match  = (rights == gold_rights) & gold_direcs
            d4o = self._train_config.loss_weight.direct_for_orient
            if d4o == 0:
                direc_weight = gold_direcs
            elif d4o < 1:
                direc_weight = gold_direcs * d4o + (1 - d4o)
                direc_weight *= gold_exists
            else:
                direc_weight = gold_exists
            
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, top3_label_logits, label_logits)

            if self._train_config.orient_hinge_loss:
                right_loss = hinge_loss(right_logits, gold_rights, direc_weight)
                direc_loss = hinge_loss(direc_logits, gold_direcs, None)
                joint_loss = hinge_loss(joint_logits, gold_joints, None)
            else:
                right_loss = binary_cross_entropy(right_logits, gold_rights, direc_weight)
                direc_loss = binary_cross_entropy(direc_logits, gold_direcs, None)
                joint_loss = binary_cross_entropy(joint_logits, gold_joints, None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.joint * joint_loss + total_loss
            total_loss = self._train_config.loss_weight.orient * right_loss + total_loss
            total_loss = self._train_config.loss_weight.direct * direc_loss + total_loss
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            gs = self.global_step
            self._writer.add_scalar('Accuracy/Tag',   1 - fraction(tag_mis,    tag_weight),   gs)
            self._writer.add_scalar('Accuracy/Label', 1 - fraction(label_mis,  label_weight), gs)
            self._writer.add_scalar('Accuracy/Oriention', fraction(orient_match, gold_direcs),gs)
            self._writer.add_scalar('Accuracy/Directional', fraction(direcs == gold_direcs),  gs)
            self._writer.add_scalar('Accuracy/Joint',       fraction(joints == gold_joints),  gs)
            self._writer.add_scalar('Loss/Tag',     tag_loss,   gs)
            self._writer.add_scalar('Loss/Label',   label_loss, gs)
            self._writer.add_scalar('Loss/Orient',  right_loss, gs)
            self._writer.add_scalar('Loss/Direct',  direc_loss, gs)
            self._writer.add_scalar('Loss/Joint',  joint_loss, gs)
            self._writer.add_scalar('Loss/Total',   total_loss, gs)
            self._writer.add_scalar('Batch/SamplePerSec', batch_len / batch_time,  gs)
            self._writer.add_scalar('Batch/Length', batch_len,   gs)
            self._writer.add_scalar('Batch/Height', batch['segments'].shape[0], gs)
        else:
            vis, _, _, pending_heads = self._vis_mode
            if pending_heads:
                b_head = tuple(batch[x] for x in 'segments seq_len token tag label right joint direc'.split())
                b_data = (segment, seq_len, tags, labels, rights, joints, direcs)
                tensors = b_head + b_data
            else:
                tensors = (segment, seq_len, batch['token'], tags, labels, rights, joints, direcs)
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
            else:
                folder = ds_name + '_test_with_devel'
        else:
            head_trees = devel_head_batchess
            folder = ds_name + '_devel'
        vis = DiscoVis(epoch,
                       self.recorder.create_join(folder),
                       self.i2vs,
                       head_trees,
                       self.recorder.log,
                       self._evalb_lcfrs_kwargs)
        pending_heads = vis._pending_heads
        vis = VisRunner(vis, async_ = False) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, pending_heads

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
            mode = 'TestSet' if use_test_set else 'DevelSet'
            self._writer.add_scalar(f'{mode}/F1',      scores['TF'], self.global_step)
            self._writer.add_scalar(f'{mode}/Disc.F1', scores['DF'], self.global_step)
            self._writer.add_scalar(f'{mode}/SamplePerSec', speed, self.global_step)
        self._vis_mode = None
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores['TF']
        return scores


from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from data.cross import disco_tree, bracketing, Counter
from data.cross.evalb_lcfrs import DiscoEvalb
def batch_trees(heads_gen, segments, i2vs, fall_back_root_label = None,
                unlabel = None,
                equal_labels = None,
                equal_words = None,
                excluded_words = None,
                excluded_labels = None):
    trees = []
    for s_seq_len, s_token, s_tag, s_label, s_right, s_joint, s_direc in heads_gen:
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

        try:
            bottom, td, rt = disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, fall_back_root_label)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(err)
            import pdb; pdb.set_trace()
            bottom, td, rt = disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, fall_back_root_label)
            brackets_cnt = None
            bottom = ((bid, wd, tg) for bid, (wd, tg) in enumerate(zip(words, tags)))
        else:
            if td:
                brackets_cnt = bracketing(bottom, td, rt, False, unlabel, excluded_labels, equal_labels)
            else:
                brackets_cnt = Counter()
        bottom_set = set()
        for bid, word, tag in bottom:
            if equal_words:
                word = equal_words.get(word, word)
            if word in excluded_words:
                continue
            bottom_set.add((bid, word, tag))
        trees.append((brackets_cnt, bottom_set))
    return trees

class DiscoVis(BaseVis):
    def __init__(self, epoch, work_dir, i2vs, head_trees, logger, evalb_lcfrs_kwargs):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._evalb = DiscoEvalb()
        self._i2vs = i2vs
        self._logger = logger
        self._head_batches = head_trees
        self._pending_heads = not head_trees
        self._data_batch_cnt = 0
        self._evalb_lcfrs_kwargs = evalb_lcfrs_kwargs

    def _before(self):
        pass

    def _process(self, batch_id, batch):

        if self._pending_heads:
            (h_segment, h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc,
             d_segment, d_seq_len,          d_tag, d_label, d_right, d_joint, d_direc) = batch
            heads = zip(h_seq_len, h_token, h_tag, h_label, h_right, h_joint, h_direc)
            heads = batch_trees(heads, h_segment, self._i2vs, **self._evalb_lcfrs_kwargs)
            self._head_batches.append(heads)
        else:
            (d_segment, d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc) = batch
            heads = self._head_batches[self._data_batch_cnt]
            self._data_batch_cnt += 1
        
        data = zip(d_seq_len, h_token, d_tag, d_label, d_right, d_joint, d_direc)
        data = batch_trees(data, d_segment, self._i2vs, 'VROOT', **self._evalb_lcfrs_kwargs)
        for gold, prediction in zip(heads, data):
            self._evalb.add(*prediction, *gold)

    def _after(self):
        num_errors = self._evalb._missing
        if num_errors:
            self._logger(f'  {num_errors} errors from evalb')
        tp, tr, tf, dp, dr, df = self._evalb.summary()
        scores = dict(TP = tp, TR = tr, TF = tf, DP = dp, DR = dr, DF = df, N = self._evalb._total_sents)
        with open(join(self._work_dir, f'eval.{self.epoch}.rpt'), 'w') as fw:
            fw.write(str(self._evalb))

        desc = f'Evalb({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
        return scores, desc, f'N: {scores["N"]} {desc}', self._head_batches
