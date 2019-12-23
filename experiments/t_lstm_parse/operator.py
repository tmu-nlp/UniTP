import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from utils.operator import Operator
from data.delta import get_rgt, get_dir, s_index
from data.penn_types import C_ABSTRACT
from time import time
from math import exp
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN
from models.utils import PCA

class PennOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, evalb):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._evalb = evalb
        self._softmax = nn.Softmax(dim = 2)
        self._sigmoid = nn.Sigmoid()
        self._orient_hinge_loss = True

    def _build_optimizer(self):
        self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.5
        self._writer = SummaryWriter(self.recorder.create_join('train'))
        self._current_scores = None
        self._last_wander_ratio = 0
        self._base_lr = 0.001
        self._lr_discount_rate = 0.001
        # for params in self._model.parameters():
        #     if len(params.shape) > 1:
        #         nn.init.xavier_uniform_(params)
        return optim.Adam(self._model.parameters(), betas = (0.9, 0.98), weight_decay = 5e-4)

    def _schedule(self, epoch, wander_ratio):
        wander_threshold = 0.15
        # lr_half_life = 40

        if wander_ratio < wander_threshold:
            learning_rate = self._base_lr * (1 - exp(- epoch))
        else:
            lr_discount = self._base_lr * self._lr_discount_rate
            if abs(self._last_wander_ratio - wander_ratio) > 1e-10: # change
                self._last_wander_ratio = wander_ratio
                if self._base_lr > lr_discount + 1e-10:
                    self._base_lr -= lr_discount
                else:
                    self._base_lr *= self._lr_discount_rate

            # if epoch > lr_half_life:
            #     base_lr *= exp(-(epoch - lr_half_life) / lr_half_life) # fine decline
            linear_dec = (1 - (wander_ratio - wander_threshold) / (1 - wander_threshold + 1e-20))
            learning_rate = self._base_lr * linear_dec
        learning_rate += 1e-20
        
        self._writer.add_scalar('Batch/Learning_Rate', learning_rate, self.global_step)
        self._writer.add_scalar('Batch/Epoch', epoch, self.global_step)
        for opg in self.optimizer.param_groups:
            opg['lr'] = learning_rate

    def _step(self, mode, ds_name, batch, flush = True, extra = None):
        # assert ds_name == C_ABSTRACT
        if mode == M_TRAIN and flush:
            self.optimizer.zero_grad()

        batch_time = time()
        gold_orients = get_rgt(batch['xtype'])
        if mode == M_TRAIN:
            batch['supervised_orient'] = gold_orients
            #(batch['offset'], batch['length'])
            
        (batch_size, batch_len, static, dynamic, layers_of_base, existences, orient_logits, tag_logits, label_logits,
         trapezoid_info) = self._model(batch['word'], **batch)
        batch_time = time() - batch_time
        
        orient_logits.squeeze_(dim = 2)
        existences   .squeeze_(dim = 2)
        if self._orient_hinge_loss:
            orients = orient_logits > 0
        else:
            orient_logits = self._sigmoid(orient_logits)
            orients = orient_logits > 0.5

        if mode == M_TRAIN:
            tags    = torch.argmax(tag_logits,   dim = 2)
            labels  = torch.argmax(label_logits, dim = 2)
            bottom_existence = existences[:, -batch_len:]
            orient_weight = get_dir(batch['xtype'])
            tag_mis       = (tags    != batch['tag'])
            label_mis     = (labels  != batch['label'])
            orient_match  = (orients == gold_orients) & orient_weight
            tag_weight    = (   tag_mis | bottom_existence)
            label_weight  = ( label_mis | existences)

            tag_loss    = cross_entropy(tag_logits,   batch['tag'],   None)
            label_loss  = cross_entropy(label_logits, batch['label'], None, batch_len, batch['length'])
            if self._orient_hinge_loss:
                orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
            else:
                orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

            total_loss = (tag_loss, label_loss, orient_loss)
            total_loss = zip(self._loss_weights_of_tag_label_orient, total_loss)
            total_loss = sum(w * loss for w, loss in total_loss)
            total_loss.backward()
            # check = existences == (batch['xtype'] > 0)
            if flush:
                self.optimizer.step()
            gs = self.global_step
            self._writer.add_scalar('Accuracy/Tag',     1 - fraction(tag_mis,    tag_weight),   gs)
            self._writer.add_scalar('Accuracy/Label',   1 - fraction(label_mis,  label_weight), gs)
            self._writer.add_scalar('Accuracy/Orient',  fraction(orient_match, orient_weight),  gs)
            self._writer.add_scalar('Loss/Tag',     tag_loss,    gs)
            self._writer.add_scalar('Loss/Label',   label_loss,  gs)
            self._writer.add_scalar('Loss/Orient',  orient_loss, gs)
            self._writer.add_scalar('Loss/Total',   total_loss,  gs)
            self._writer.add_scalar('Batch/SamplePerSec', batch_len / batch_time,  gs)
            self._writer.add_scalar('Batch/Length', batch_len,   gs)
        else:
            vis, batch_id = extra
            mpc_word = mpc_label = None
            if vis.save_tensors:
                if hasattr(self._model._emb_layer, 'pca'):
                    if dynamic is not None: # even dynamic might be None, being dynamic is necessary to train a good model
                        mpc_word = self._model._emb_layer.pca(static, flush = flush)
                    mpc_label    = self._model._emb_layer.pca(layers_of_base)
                else:
                    mpc_label = PCA(layers_of_base[:, -batch_len:].reshape(-1, layers_of_base.shape[2]))(layers_of_base)

                tag_logits   = self._softmax(tag_logits)
                label_logits = self._softmax(label_logits)
                tag_scores,   tags   = tag_logits  .topk(1)
                label_scores, labels = label_logits.topk(1)
                tags  .squeeze_(dim = 2)
                labels.squeeze_(dim = 2)
                tag_scores  .squeeze_(dim = 2)
                label_scores.squeeze_(dim = 2)
                if self._orient_hinge_loss: # otherwise with sigmoid
                    orient_logits += 1
                    orient_logits /= 2
                    orient_logits[orient_logits < 0] = 0
                    orient_logits[orient_logits > 1] = 1
                b_mpcs = (None if mpc_word is None else mpc_word.type(torch.float16), mpc_label.type(torch.float16))
                b_scores = (tag_scores.type(torch.float16), label_scores.type(torch.float16), orient_logits.type(torch.float16))
            else:
                tags   = torch.argmax(tag_logits,   dim = 2)
                labels = torch.argmax(label_logits, dim = 2)
                b_mpcs = (mpc_word, mpc_label)
                b_scores = (None, None, None)
            b_size = (batch_len,)
            b_head = tuple(batch[x].type(torch.uint8) if x in ('tag', 'label') else batch[x] for x in 'offset length word tag label'.split())
            b_head = b_head + (gold_orients,)
            b_logits = (tags.type(torch.uint8), labels.type(torch.uint8), orients)
            b_data = b_logits + b_mpcs + b_scores
            if trapezoid_info is not None:
                trapezoid_info = (batch['segment'], batch['seg_length']) + trapezoid_info
            vis._process(batch_id, b_size + b_head + b_data, trapezoid_info)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False):
        def set_scores(scores):
            self._current_scores = scores
        folder = 'vis_test' if use_test_set else 'vis_devel'
        self._model.eval()
        return PennTrainVis(epoch,
                            self.recorder.create_join(folder),
                            self._evalb,
                            self.i2vs,
                            self.recorder.log,
                            set_scores,
                            is_bin_times(int(float(epoch)) - 1))

    def _after_validation(self, vis):
        self._model.train()
        # vis.wait()

    def _key(self):
        return self._current_scores['key']

    def _scores(self):
        return self._current_scores

def fraction(cnt_n, cnt_d, dtype = torch.float32):
    return cnt_n.sum().type(dtype) / cnt_d.sum().type(dtype)

def cross_entropy(x_, y_, w_, *s_l_):
    b_, t_, c_ = x_.shape
    losses = F.cross_entropy(x_.view(-1, c_), y_.view(-1), reduction = 'none')
    if w_ is not None:
        losses = losses * w_.view(-1)
    if s_l_:
        s_, l_ = s_l_
        losses = losses * height_mask(t_, s_, l_).view(-1)
    return losses.sum() # TODO turn off non-train gradient tracking

def height_mask(flatten_triangle_size, bottom_size, bottom_lengths):
    p_ = torch.arange(flatten_triangle_size, device = bottom_lengths.device)[None, :, None]
    t_ = s_index(bottom_size - bottom_lengths)[:, None, None]
    return p_ >= t_

def binary_cross_entropy(x, y, w):
    losses = F.binary_cross_entropy(x, y.type(x.dtype), reduction = 'none')
    losses = losses * w
    return losses.sum()

def hinge_loss(x, y, w):
    ones = torch.ones_like(x)
    y = torch.where(y, ones, -ones)
    losses = 1 - (x * y)
    losses[(losses < 0) | ~ w] = 0
    return losses.sum()
    

# from utils.vis import Vis
from utils.file_io import join, isfile, listdir, remove
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from visualization import set_vocab, set_head, set_data
class PennTrainVis:#(Vis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger, set_scores, save_tensors = True):
        # super().__init__(epoch)
        self.epoch = epoch
        self._work_dir = work_dir
        self._evalb = evalb
        self._i2vs = i2vs
        self._logger = logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{self.epoch}.tree')
        self._fnames = htree, dtree
        self._head_tree = None
        self._data_tree = None
        self._set_scores = set_scores
        self._save_tensors = save_tensors

    @property
    def save_tensors(self):
        return self._save_tensors

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _before(self):
        # overwrite
        htree, dtree = self._fnames
        if set_vocab(self._work_dir, self._i2vs._nested):
            # for fname in listdir(self._work_dir):
            #     if fname.startswith('head.'):
            #         fname = join(self._work_dir, fname)
            #         remove(fname)
            #     if fname.startswith('data.'):
            self._head_tree = open(htree, 'w')
        self._data_tree = open(dtree, 'w')

    def _process(self, batch_id, batch, trapezoid_info):
        # process batch to instances, compress
        # if head is in batch, save it
        # else check head.emm.pkl
        # make data.emmb.tree & concatenate to data.emm.tree
        # make data.emm.rpt
        batch = (x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in batch)
        (size, h_offset, h_length, h_word, h_tag, h_label, h_right, 
         d_tag, d_label, d_right, mpc_word, mpc_label,
         tag_score, label_score, split_score) = batch
        if trapezoid_info:
            segment, seg_length, d_segment, d_seg_length = trapezoid_info
            trapezoid_info = segment, seg_length
            d_trapezoid_info = d_segment, d_seg_length.cpu().numpy()
        if self._head_tree:
            set_head(self._work_dir, batch_id,
                     size, h_offset, h_length, h_word, h_tag, h_label, h_right,
                     trapezoid_info,
                     self._i2vs, self._head_tree)
        
        fpath = self._work_dir if self._save_tensors else None
        set_data(fpath, batch_id, size, self.epoch, 
                 h_offset, h_length, h_word, d_tag, d_label, d_right,
                 mpc_word, mpc_label,
                 tag_score, label_score, split_score,
                 d_trapezoid_info if trapezoid_info else None,
                 self._i2vs, self._data_tree, self._logger, self._evalb)

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        proc = parseval(self._evalb, *self._fnames)
        scores = rpt_summary(proc.stdout.decode(), False, True)
        errors = proc.stderr.decode()
        num_errors = sum(len(x) for x in errors.split('\n') if len(x))
        if num_errors > 10:
            self._logger(f'  {num_errors} errors from evalb')
        elif num_errors:
            self._logger(errors, end = '')
        self._head_tree = self._data_tree = None
        scores['key'] = scores.get('F1', 0)
        self._set_scores(scores)
        return ', '.join(f'{k}: {v}' for k,v in scores.items())