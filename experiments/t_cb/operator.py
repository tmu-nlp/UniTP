from data.stan_types import C_SSTB
from data.continuous.binary import X_RGT, X_DIR
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, K_CORP, F_CNF, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06, make_close_frac
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss, cross_entropy, get_label_height_mask, sorted_decisions_with_values
from experiments.helper import make_tensors, continuous_score_desc_logg, sentiment_score_desc_logg
from experiments.helper.co import CO, CVP
from time import time
import torch

train_type = dict(loss_weight = dict(tag    = BaseType(0.2, validator = frac_open_0),
                                     label  = BaseType(0.3, validator = frac_open_0),
                                     orient = BaseType(0.5, validator = frac_open_0),
                                     polar  = BaseType(0.9, validator = frac_open_0),
                                     polaro = make_close_frac(0.1)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  orient_hinge_loss = true_type,
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))

class CBOperator(CO):
    def _step(self, mode, ds_name, batch, batch_id = None):

        batch['key'] = corp = ds_name if self.multi_corp else None
        batch['ignore_logits'] = within_structure = ds_name == C_SSTB
        if mode == M_TRAIN or within_structure:
            batch['supervision'] = gold_orients = (X_RGT & batch['xtype']) > 0

        batch_time = time()
        bottom, stem, tag_label_or_polar = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time
        batch_size, batch_len = bottom[:2]
        existences = stem.existence
        orient_logits, batch_segment = stem.extension

        orient_logits.squeeze_(dim = 2)
        existences   .squeeze_(dim = 2)
        orients = self._model.stem.orientation(orient_logits)
        if not self._train_config.orient_hinge_loss:
            orient_logits = self._sigmoid(orient_logits)

        if mode == M_TRAIN:
            orient_weight = (X_DIR & batch['xtype']) > 0
            orient_match  = (orients == gold_orients) & orient_weight
            if self._train_config.orient_hinge_loss:
                orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
            else:
                orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

            if ds_name != C_SSTB:
                tag_start, tag_end, tag_logits, label_logits, _ = tag_label_or_polar
                tags   = self._model.get_decision(tag_logits)
                labels = self._model.get_decision(label_logits)
                tag_mis      = (   tags != batch['tag'])
                label_mis    = ( labels != batch['label'])
                tag_weight   = (   tag_mis | existences[:, tag_start:tag_end])
                label_weight = ( label_mis | existences[:, tag_start:])
                tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, label_weight, corp)
                total_loss = self._train_config.loss_weight.orient * orient_loss
                total_loss = self._train_config.loss_weight.tag * tag_loss + total_loss
                total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            else:
                polar_logits = tag_label_or_polar
                polars = polar_logits.argmax(dim = 2)
                polar_mis    = (polars != batch['polar'])
                polar_weight = (polar_mis | existences)
                polar_weight &= get_label_height_mask(batch, 'polar')
                polar_loss = cross_entropy(polar_logits, batch['polar'], polar_weight)
                total_loss = self._train_config.loss_weight.polar * polar_loss
                if self._train_config.loss_weight.polaro:
                    total_loss = self._train_config.loss_weight.polaro * orient_loss + total_loss

            total_loss.backward()
            
            if self.recorder._writer is not None:
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
                if ds_name != C_SSTB:
                    self.recorder.tensorboard(self.global_step, 'Accuracy/%s', ds_name,
                        Tag    = 1 - fraction(tag_mis,     tag_weight),
                        Label  = 1 - fraction(label_mis, label_weight),
                        Orient = fraction(orient_match, orient_weight))
                    self.recorder.tensorboard(self.global_step, 'Loss/%s', ds_name,
                        Tag    = tag_loss,
                        Label  = label_loss,
                        Orient = orient_loss,
                        Total  = total_loss)
                else:
                    self.recorder.tensorboard(self.global_step, 'Accuracy/%s', ds_name,
                        Polar  = 1 - fraction(polar_mis, polar_weight),
                        Orient = fraction(orient_match, orient_weight))
                    self.recorder.tensorboard(self.global_step, 'Loss/%s', ds_name,
                        Polar  = polar_loss,
                        Orient = orient_loss,
                        Total  = total_loss)
                self.recorder.tensorboard(self.global_step, 'Batch/%s', ds_name,
                    Length = batch_len,
                    Height = len(stem.segment),
                    SamplePerSec = batch_len / batch_time)
        else:
            vis, _, _, serial = self._vis_mode
            b_head = [batch['tree'], batch.get('offset'), batch['length'], batch['token']]
            if serial:
                if (pca := (self._model.get_static_pca(corp) if hasattr(self._model, 'get_static_pca') else None)) is None:
                    pca = PCA(stem.embedding.reshape(-1, stem.embedding.shape[2]))
                b_head += [pca(bottom.embedding).type(torch.float16), pca(stem.embedding).type(torch.float16)]
                if self._train_config.orient_hinge_loss: # otherwise with sigmoid
                    hinge_score(orient_logits, inplace = True)

                if ds_name != C_SSTB:
                    tag_scores,   tags   = self._model.get_decision_with_value(tag_label_or_polar.tag)
                    label_scores, labels = self._model.get_decision_with_value(tag_label_or_polar.label)
                    b_data = [tags.type(torch.short), labels.type(torch.short), orients]
                    b_data += [tag_scores.type(torch.float16), label_scores.type(torch.float16), orient_logits.type(torch.float16)]
                else:
                    polar_scores, polars = sorted_decisions_with_values(self._softmax, 5, tag_label_or_polar)
                    b_data = [None, polars.type(torch.uint8), gold_orients, None, polar_scores.type(torch.float16), orient_logits.type(torch.float16)]
            else:
                if ds_name != C_SSTB:
                    tags   = self._model.get_decision(tag_label_or_polar.tag  )
                    labels = self._model.get_decision(tag_label_or_polar.label)
                    b_data = [tags.type(torch.short), labels.type(torch.short), orients]
                else:
                    _, polars = sorted_decisions_with_values(self._softmax, 5, tag_label_or_polar)
                    b_data = [None, polars.type(torch.uint8), gold_orients]

            tensors = make_tensors(batch_id, batch_size, batch_len, *b_head, *b_data)
            vis.process(tensors, None if batch_segment is None else (stem.segment, batch_segment.cpu().numpy()))
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_bins, test_bins = self._mode_length_bins
        devel_init, test_init = self._initial_run
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            if final_test:
                folder = ds_name + '_test'
                scores_of_bins = save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
                scores_of_bins = False
            if self.multi_corp:
                flush_heads = test_init[ds_name]
                test_init[ds_name] = False
            else:
                flush_heads = test_init
                self._initial_run = devel_init, False
            length_bins = test_bins
        else:
            folder = ds_name + '_devel'
            save_tensors = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
            scores_of_bins = False
            if self.multi_corp:
                flush_heads = devel_init[ds_name]
                devel_init[ds_name] = False
            else:
                flush_heads = devel_init
                self._initial_run = False, test_init
            length_bins = devel_bins

        if self.multi_corp:
            m_corp = ds_name
            i2vs = self.i2vs[ds_name]
            length_bins = length_bins[ds_name]
        else:
            m_corp = None
            i2vs = self.i2vs
        if hasattr(self._model, 'update_static_pca'):
            self._model.update_static_pca(m_corp)

        work_dir = self.recorder.create_join(folder)
        if serial := (save_tensors or flush_heads or self.dm is None or ds_name == C_SSTB):
            v_cls = ParsingCBVA if ds_name != C_SSTB else SentimentCBVA
            vis = v_cls(epoch,
                        work_dir,
                        self._evalb,
                        i2vs,
                        self.recorder.log,
                        save_tensors,
                        length_bins,
                        scores_of_bins,
                        flush_heads)
        else:
            vis = CBVP(epoch, work_dir, self._evalb, self.recorder.log, self.dm, m_corp)
        vis = VisRunner(vis, async_ = serial) # wrapper
        self._vis_mode = vis, use_test_set, final_test, serial

    def _get_optuna_fn(self, train_params):
        from utils.train_ops import train, get_optuna_params
        from utils.str_ops import height_ratio
        from utils.math_ops import log_to_frac
        from utils.types import F_SENTENCE, F_PHRASE

        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                new_factor = {}
                desc = []
                for corp, factor in specs['data'][K_CORP].items():
                    if corp == C_SSTB:
                        continue
                    desc_ = [corp + '.']
                    level, left, _ = factor['binarization'].split()
                    if level == F_SENTENCE and left == F_CNF:
                        binarization = trial.suggest_float(corp + '.cnf', 0.0, 1.0)
                        factor['binarization'] = f'{level} {left} {binarization}'
                        desc_.append(height_ratio(binarization))
                        binarization = level, F_CNF, binarization
                    else:
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
                self._train_materials = new_factor, self._train_materials[1] # for train/train_initials(max_epoch>0)
                if C_SSTB in specs['data'][K_CORP]:
                    loss_weight = specs['train']['loss_weight']
                    loss_weight['polar' ] = pl = trial.suggest_float('polar',  0, 1)
                    loss_weight['polaro'] = po = trial.suggest_float('polaro', 0, 1)
                    desc.append(C_SSTB + '.' + height_ratio(pl) + height_ratio(po))
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = new_lr = trial.suggest_float('learning_rate', 1e-5, lr, log = True)
                self._train_config._nested.update(specs['train'])
                desc.append('γ=' + height_ratio(log_to_frac(new_lr, 1e-5, lr)))
                return '.'.join(desc)

            self._init_mode_trees()
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from data.mp import BaseVis, VisRunner
from utils.file_io import join, isfile, remove
from utils.shell_io import parseval, rpt_summary
from visualization import ContinuousTensorVis
class CBVA(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 save_tensors   = True,
                 length_bins    = None,
                 scores_of_bins = False,
                 flush_heads    = False):
        super().__init__(epoch, work_dir, i2vs)
        self._evalb = evalb
        if flush_heads and isfile(fname := self.join('vocabs.pkl')):
            remove(fname)
        self._ctvis = ContinuousTensorVis(work_dir, i2vs)
        self._logger = logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._fnames = htree, dtree
        self._head_tree = None
        self._data_tree = None
        self._scores_of_bins = scores_of_bins
        self.save_tensors = save_tensors
        self._length_bins = length_bins

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _before(self):
        htree, dtree = self._fnames
        if self._ctvis.is_anew: # TODO
            self._head_tree = open(htree, 'w')
            self._length_bins = set()
        self._data_tree = open(dtree, 'w')

    def _process(self, batch, trapezoid_info):
        (batch_id, _, size, head_tree, offset, length, token, token_emb, tree_emb,
         tag, label, orient, tag_score, label_score, orient_score) = batch

        if self._head_tree:
            self._length_bins |= self._ctvis.set_head(self._head_tree, head_tree, batch_id, size, 10, length, token)

        if self.save_tensors and tag is not None:
            if self._length_bins is not None and self._scores_of_bins:
                bin_width = 10
            else:
                bin_width = None
            extra = size, bin_width, self._evalb
        else:
            extra = None

        self._ctvis.set_data(self._data_tree, self._logger, batch_id, self.epoch,
                             offset, length, token, token_emb, tree_emb,
                             tag, label, orient,
                             tag_score, label_score, orient_score,
                             trapezoid_info, extra)

from data.stan_types import calc_stan_accuracy
class ParsingCBVA(CBVA):
    def _after(self):
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        proc = parseval(self._evalb, *self._fnames)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        if num_errors := len(errors):
            self._logger(f'  {num_errors} errors from evalb')
            if num_errors < 100:
                for e, error in enumerate(errors):
                    self._logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'
                with open(self._ctvis.join(fname), 'w') as fw:
                    fw.write(report)
                self._logger(f'  (Check {fname} for details.)')

        self._head_tree = self._data_tree = None

        if self._length_bins is not None and self._scores_of_bins:
            with open(self._ctvis.join(f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self._length_bins:
                    fhead = self._ctvis.join(f'head.bin_{wbin}.tree')
                    fdata = self._ctvis.join(f'data.bin_{wbin}.tree')
                    proc = parseval(self._evalb, fhead, fdata)
                    smy = rpt_summary(proc.stdout.decode(), False, True)
                    fw.write(f"{wbin},{','.join(str(smy.get(x, 0)) for x in ('N', 'LP', 'LR', 'F1', 'TA'))}\n")
                    remove(fhead)
                    remove(fdata)

        return continuous_score_desc_logg(scores) + (self._length_bins,)

from collections import defaultdict
class SentimentCBVA(CBVA):
    def _after(self):
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        flip_lines = defaultdict(list) if self.save_tensors else None
        _, smy = calc_stan_accuracy(*self._fnames, self._logger, flip_lines)
        if flip_lines:
            for flip_n, nhd in flip_lines.items():
                nhd.sort(key = lambda x: x[0] - flip_n)
                with open(self.join(f'flip.{self.epoch}.{flip_n}.art'), 'w') as fw:
                    for eid, (n, head, data) in enumerate(nhd):
                        fw.write(f'{eid}. prediction has {n - flip_n} more flips\n\n')
                        fw.write('\n'.join(head) + '\n\n')
                        fw.write('\n'.join(data) + '\n\n')
        return sentiment_score_desc_logg(smy)


class ScatterVis(BaseVis):
    def __init__(self, epoch, work_dir, dim = 10):
        super().__init__(epoch, work_dir, None)
        self._work_dir = work_dir
        self._fname = dim


    def _before(self):
        line = 'label,' + ','.join(f'pc{i}' if i else 'mag' for i in range(self._fname)) + '\n'
        fname = join(self._work_dir, f'pca.{self.epoch}.csv')
        self._fname = fname
        with open(fname, 'w') as fw:
            fw.write(line)

    def _process(self, label_embeddings):
        with open(self._fname, 'a+') as fw:
            for label, embeddings in label_embeddings.items():
                for emb in embeddings:
                    fw.write(label + ',' + ','.join(f'{e:.3f}' for e in emb) + '\n')

    def _after(self):
        pass

class CBVP(CVP):
    def _process(self, batch, d_trapezoid_info):
        (batch_id, _, _, _, offset, length, token,
         tag, label, orient) = batch
        dm, _, _, corp_key = self._args
        if offset is None:
            offset = length * 0
        
        if d_trapezoid_info:
            segment, batch_segment = d_trapezoid_info
            dm.batch(batch_id, segment, offset, length, token, tag, label, orient, batch_segment, key = corp_key)
        else:
            dm.batch(batch_id, offset, length, token, tag, label, orient, key = corp_key)

# an example of Unmatched Length from evalb
# head
# (S (S (VP (VBG CLUBBING) (NP (DT A) (NN FAN)))) (VP (VBD was) (RB n't) (NP (NP (DT the) (NNP Baltimore) (NNP Orioles) (POS ')) (NN fault))) (. .))
# (S (NP (NP (JJ CLUBBING) (NNP A)) ('' FAN)) (VP (VBD was) (PP (RB n't) (NP     (DT the) (NNP Baltimore) (NNS Orioles) (POS ') (NN fault)))) (. .))
# data
