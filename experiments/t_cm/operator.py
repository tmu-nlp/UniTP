import torch
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06
from models.utils import fraction, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper.co import CO, CVP, tee_trees
from experiments.helper import make_tensors, speed_logg, continuous_score_desc_logg
from data.penn_types import C_PTB, E_NER


train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     chunk = BaseType(0.5, validator = frac_open_0)),
                  chunk_hinge_loss = true_type,
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))
                #   keep_low_attention_rate = BaseType(1.0, validator = frac_close),

class CMOperator(CO):
    def _step(self, mode, ds_name, batch, batch_id = None):

        batch['key'] = corp = ds_name if self.multi_corp else None
        if mode == M_TRAIN:
            batch['supervision'] = gold_chunks = batch['chunk']
            # supervised_signals['keep_low_attention_rate'] = self._train_config.keep_low_attention_rate
        elif 'tag_layer' in batch:
            batch['bottom_supervision'] = batch['tag_layer'], batch['char_chunk']

        batch_time = time()
        bottom, stem, tag_label = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time
        batch_size, batch_len = bottom[:2]
        existences = stem.existence
        tag_start, tag_end, tag_logits, label_logits, _ = tag_label
        weight, chunk_logits, chunk_vote, batch_segment = stem.extension

        chunks = self._model.stem.chunk(chunk_logits)
        if not self._train_config.chunk_hinge_loss:
            chunk_logits = self._sigmoid(chunk_logits)

        if ds_name in E_NER:
            if batch['n_layers'] == 1:
                label_logits = label_logits[:, tag_end:].contiguous()
                label_existence = existences[:, tag_end:]
            else: # bio 1-1 labelling
                label_logits = label_logits[:, :tag_end].contiguous()
                label_existence = existences[:, :tag_end]
        else:
            label_existence = existences[:, tag_start:]

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            tag_mis      = (tags    != batch['tag'  ])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | existences[:, tag_start: tag_end])
            label_weight = (label_mis | label_existence)
            
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, label_weight, corp, height_mask = ds_name not in E_NER)
            if self._train_config.chunk_hinge_loss:
                chunk_loss = hinge_loss(chunk_logits, gold_chunks, None)
            else:
                chunk_loss = binary_cross_entropy(chunk_logits, gold_chunks, None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.chunk * chunk_loss + total_loss
            total_loss.backward()
            
            if self.recorder._writer is not None:
                suffix = ds_name if self.multi_corp else None
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
                self.recorder.tensorboard(self.global_step, 'Accuracy/%s', suffix,
                    Tag   = 1 - fraction(tag_mis,     tag_weight),
                    Label = 1 - fraction(label_mis, label_weight),
                    Fence = fraction(chunks == gold_chunks))
                self.recorder.tensorboard(self.global_step, 'Loss/%s', suffix,
                    Tag   = tag_loss,
                    Label = label_loss,
                    Fence = chunk_loss,
                    Total = total_loss)
                self.recorder.tensorboard(self.global_step, 'Batch/%s', suffix,
                    Length = batch_len,
                    Height = len(stem.segment),
                    SamplePerSec = batch_len / batch_time)
        else:
            vis, _, _, serial, draw_weights = self._vis_mode
            b_head = [batch['tree'], batch['length'], batch['token']]
            tags   = self._model.get_decision(tag_logits  )
            labels = self._model.get_decision(label_logits)
            b_data = [tags.type(torch.short), labels.type(torch.short), chunks, stem.segment, batch_segment, batch.get('tag_layer', 0)]
            if ds_name in E_NER:
                b_data += [tag_end]
            elif serial: # [tree, length, token, tag, label, chunk, b_seg, segment, weight, vote]
                if draw_weights:
                    b_data.append(mean_stdev(weight).type(torch.float16))
                    b_data.append(None if chunk_vote is None else chunk_vote.type(torch.float16))
                else:
                    b_data += [None, None]
            vis.process(batch_id, make_tensors(*b_head, *b_data))
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_bins, test_bins = self._mode_length_bins
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            if final_test:
                folder = ds_name + '_test'
                draw_weights = True
            else:
                folder = ds_name + '_test_with_devel'
                if int(epoch_minor) == 0:
                    draw_weights = is_bin_times(int(epoch_major))
                else:
                    draw_weights = False
            length_bins = test_bins
        else:
            folder = ds_name + '_devel'
            if int(epoch_minor) == 0:
                draw_weights = is_bin_times(int(epoch_major))
            else:
                draw_weights = False
            length_bins = devel_bins

        m_corp = None
        i2vs = self.i2vs
        if self.multi_corp:
            m_corp = ds_name
            i2vs = i2vs[ds_name]
            length_bins = length_bins[ds_name]
        work_dir = self.recorder.create_join(folder)
        
        if serial := (draw_weights or length_bins is None or self.dm is None or ds_name in E_NER):
            if ds_name in E_NER:
                vis = NERVA(
                    epoch,
                    work_dir,
                    self._evalb,
                    i2vs,
                    self.recorder.log,
                    length_bins is None
                )
            else:
                vis = CMVA(
                    epoch,
                    work_dir,
                    self._evalb,
                    i2vs,
                    self.recorder.log,
                    ds_name == C_PTB,
                    draw_weights,
                    length_bins
                )
        else:
            vis = CMVP(epoch, work_dir, self._evalb, self.recorder.log, self.dm, m_corp)
        vis = VisRunner(vis, async_ = serial) # wrapper
        self._vis_mode = vis, use_test_set, final_test, serial, draw_weights


    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial, _ = self._vis_mode
        devel_bins, test_bins = self._mode_length_bins
        if serial:
            scores, desc, logg, length_bins = vis.after()
        else:
            scores, desc, logg = vis.after()
            length_bins = True
        if self.multi_corp:
            desc = ds_name.upper() + desc
            logg = ds_name + ' ' + logg
        else:
            desc = 'Evalb' + desc

        if self.multi_corp:
            if use_test_set:
                test_bins [ds_name] = length_bins
            else:
                devel_bins[ds_name] = length_bins
        else:
            if use_test_set:
                self._mode_length_bins = devel_bins, length_bins # change test
            else:
                self._mode_length_bins = length_bins, test_bins # change devel

        _desc, _logg, speed_outer, speed_dm = speed_logg(count, seconds, None if serial else self._dm)
        scores['speed'] = speed_outer
        if not final_test:
            prefix = 'TestSet' if use_test_set else 'DevelSet'
            suffix = ds_name if self.multi_corp else None
            self.recorder.tensorboard(self.global_step, prefix + '/%s', suffix,
                                      F1 = scores.get('F1', 0),
                                      SamplePerSec = None if serial else speed_dm)
        return scores, desc + _desc, logg + _logg

    def _get_optuna_fn(self, train_params):
        from utils.train_ops import train, get_optuna_params
        from utils.str_ops import height_ratio
        from utils.math_ops import log_to_frac
        from utils.types import K_CORP

        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                new_factor = {}
                desc = []
                for corp, factor in specs['data'][K_CORP].items():
                    factor['esub'] = esub = trial.suggest_float(corp + '.e', 0.0, 1.0)
                    if msub := factor['msub']:
                        factor['msub'] = msub = trial.suggest_float(corp + '.m', 0.0, 1.0)
                    new_factor[corp] = esub, msub
                    desc.append(corp + '.∅' + height_ratio(esub) + height_ratio(msub))
                self._train_materials = new_factor, self._train_materials[1]
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
from utils.file_io import join, isfile, listdir, remove
from utils.shell_io import parseval, rpt_summary
from data.continuous.multib.mp import tensor_to_tree
from data.continuous import draw_str_lines
from sys import stderr
from copy import deepcopy

class CMVA(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 mark_np_without_dt,
                 draw_weights   = False,
                 length_bins    = None):
        super().__init__(epoch, work_dir, i2vs)
        self._evalb = evalb
        htree = self.join('head.tree')
        dtree = self.join(f'data.{epoch}.tree')
        self._fnames   = htree, dtree
        self._is_anew  = not isfile(htree)
        self._rpt_file = self.join(f'data.{epoch}.rpt')
        self._logger   = logger
        self._i2vs     = i2vs.token, i2vs.tag, i2vs.label, 
        self._length_bins = length_bins
        self._draw_file = self.join(f'data.{epoch}.art') if draw_weights else None
        self._error_idx = 0, []
        self._headedness_stat = self.join(f'data.{epoch}.headedness'), {}
        self._mark_np_without_dt = mark_np_without_dt

    def _before(self):
        htree, dtree = self._fnames
        if isfile(dtree): remove(dtree)
        if self._is_anew:
            self._length_bins = set()
            if isfile(htree): remove(htree)
            for fname in listdir(self._work_dir):
                if 'bin_' in fname and fname.endswith('.tree'):
                    if fname.startswith('head.') or fname.startswith('data.'):
                        remove(self.join(fname))
        if self._draw_file and isfile(self._draw_file):
            remove(self._draw_file)

    def _process(self, _, batch):
        (trees, length, token, tag, label, chunk, batch_segment, segment, tag_layer, weight, vote) = batch

        if self._is_anew:
            str_trees = []
            for tree in trees:
                tree = deepcopy(tree)
                tree.un_chomsky_normal_form()
                str_trees.append(' '.join(str(tree).split()))
            self._length_bins |= tee_trees(self.join, 'head', length, str_trees, None, 10)

        str_trees = []
        idx_cnt, error_idx = self._error_idx
        a_args = self._i2vs + (tag_layer, batch_segment,)
        b_args = token, tag, label, chunk, segment
        for args in zip(*b_args):
            tree, safe = tensor_to_tree(*a_args, *args, fallback_label = 'VROOT')
            tree.un_chomsky_normal_form()
            idx_cnt += 1 # start from 1
            if not safe:
                error_idx.append(idx_cnt)
            str_trees.append(' '.join(str(tree).split()))
        self._error_idx = idx_cnt, error_idx

        if self._draw_file is None:
            bin_size = None
        else:
            bin_size = None if self._length_bins is None else 10
            _, head_stat = self._headedness_stat
            with open(self._draw_file, 'a+') as fw:
                for eid, args in enumerate(zip(*b_args)):
                    tree, safe, stat = tensor_to_tree(*a_args, *args, 
                        None if weight is None else weight[eid], 
                        None if   vote is None else   vote[eid],
                        fallback_label           = 'VROOT',
                        mark_np_without_dt_child = self._mark_np_without_dt)
                    for lb, (lbc, hc) in stat.items():
                        if lb in head_stat:
                            label_cnt, head_cnts = head_stat[lb]
                            for h, c in hc.items():
                                head_cnts[h] += c
                            head_stat[lb] = lbc + label_cnt, head_cnts
                        else:
                            head_stat[lb] = lbc, hc
                    if not safe:
                        fw.write('\n[*]\n')
                    try:
                        fw.write('\n'.join(draw_str_lines(tree)) + '\n\n')
                    except Exception as err:
                        print('  FAILING DRAWING:', err, file = stderr)
                        fw.write('FAILING DRAWING\n\n')
        tee_trees(self.join, f'data.{self.epoch}', length, str_trees, None, bin_size)

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        proc = parseval(self._evalb, *self._fnames)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        idx_cnt, error_idx = self._error_idx
        error_cnt = len(error_idx)
        if error_cnt < 50:
            self._logger(f'  {error_cnt} conversion errors')
        else:
            self._logger(f'  {error_cnt} conversion errors: ' + ' '.join(str(x) for x in error_idx))
        if num_errors:
            self._logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    self._logger(f'    {e}. ' + error)
        with open(self._rpt_file, 'w') as fw:
            fw.write(report)
            if num_errors >= 10:
                self._logger(f'  (Check {self._rpt_file} for details.)')
                fw.write('\n\n' + '\n'.join(errors))

        if self._length_bins is not None and self._draw_file is not None:
            with open(self.join(f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self._length_bins:
                    fhead = self.join(f'head.bin_{wbin}.tree')
                    fdata = self.join(f'data.{self.epoch}.bin_{wbin}.tree')
                    proc = parseval(self._evalb, fhead, fdata)
                    smy = rpt_summary(proc.stdout.decode(), False, True)
                    fw.write(f"{wbin},{','.join(str(smy.get(x, 0)) for x in ('N', 'LP', 'LR', 'F1', 'TA'))}\n")

        fname, head_stat = self._headedness_stat
        with open(fname, 'w') as fw:
            for label, (label_cnt, head_cnts) in sorted(head_stat.items(), key = lambda x: x[1][0], reverse = True):
                line = f'{label}({label_cnt})'.ljust(15)
                for h, c in sorted(head_cnts.items(), key = lambda x: x[1], reverse = True):
                    line += f'{h}({c}); '
                fw.write(line[:-2] + '\n')

        return continuous_score_desc_logg(scores) + (self._length_bins,)

class CMVP(CVP):
    def _process(self, batch_id, batch):
        (_, _, token, tag, label, chunk, batch_segment, segment, tag_layer) = batch
        dm, _, _, corp_key = self._args
        dm.batch(batch_id, tag_layer, batch_segment, token, tag, label, chunk, segment, key = corp_key)


import numpy as np
from data.ner_types import bio_to_tree, ner_to_tree, recover_bio_prefix
def batch_trees(b_length, b_word, b_tag, b_label, b_fence, i2vs, b_weight = None, **kw_args):
    has_tag = b_tag is not None
    has_bio = b_fence is None
    for sid, (ln, word) in enumerate(zip(b_length, b_word)):
        wd = [i2vs.token[i] for i in word [     :ln]]
        ps = [i2vs  .tag[i] for i in b_tag[sid, :ln]] if has_tag else None
        
        if has_bio:
            bi = [i2vs.label[i] for i in b_label[sid, :ln]]
            yield bio_to_tree(wd, bi, ps, **kw_args), (wd, bi)
        else:
            nr = [i2vs.label[i] for i in b_label[sid, :b_fence[sid, 1:].sum()]]
            fence, = np.where(b_fence[sid]) #.nonzero() in another format
            weights = b_weight[sid, :ln] if b_weight is not None else None
            bi = recover_bio_prefix(fence, nr)
            yield ner_to_tree(wd, nr, fence, ps, weights = weights, **kw_args), (wd, bi)

from data.continuous import draw_str_lines
from utils.file_io import join, isfile, remove
from utils.shell_io import parseval, rpt_summary
from nltk.tree import Tree
class NERVA(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger, flush_heads = False):
        super().__init__(epoch, work_dir, i2vs)
        self._evalb_i2vs_logger = evalb, i2vs, logger
        htree = self.join('head.tree')
        dtree = self.join(f'data.{epoch}.tree')
        self._art_lines = []
        self._art = self.join(f'data.{epoch}.art')
        self._err = self.join(f'data.{epoch}.rpt')
        self._fnames = htree, dtree
        self._head_tree = flush_heads
        self._data_tree = None

    def _before(self):
        htree, dtree = self._fnames
        if self._head_tree:
            self._head_tree = open(htree, 'w')
        self._data_tree = open(dtree, 'w')

    def __del__(self):
        self.close()

    def close(self):
        if self._head_tree and not isinstance(self._head_tree, bool): self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _process(self, _, batch):
        _, i2vs, _ = self._evalb_i2vs_logger
        (trees, length, token, tag, label, chunk, batch_segment, segment, tag_layer, tag_end) = batch
        if self._head_tree:
            for tree in trees:
                print(' '.join(str(tree).split()), file = self._head_tree)

        for tree, _ in batch_trees(length, token, tag, label, chunk, i2vs):
            print(' '.join(str(tree).split()), file = self._data_tree)
            self._art_lines.append('\n'.join(draw_str_lines(tree)))

    def _after(self): # TODO TODO TODO length or num_ners
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        evalb, _, logger = self._evalb_i2vs_logger
        proc = parseval(evalb, *self._fnames)
        report = proc.stdout.decode()
        s_rows, scores = rpt_summary(report, True, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        if num_errors:
            logger(f'  {num_errors} errors from evalb')
            for e, error in enumerate(errors[:10]):
                logger(f'    {e}. ' + error)
            if num_errors > 10:
                logger('     ....')
            logger(f'  (Check {self._err} for details.)')
        with open(self._err, 'w') as fw:
            fw.write(report)

        self._head_tree = self._data_tree = None
        with open(self._fnames[0]) as fr, open(self._art, 'w') as fw:
            for gold_tree, pred_tree, s_row in zip(fr, self._art_lines, s_rows):
                if not s_row:
                    breakpoint()
                lines = f'Sent #{s_row[0]}:'
                if s_row[3] == s_row[4] == 0 or s_row[3] == s_row[4] == 100:
                    lines += ' EXACT MATCH\n' + pred_tree + '\n\n\n'
                else:
                    gold_tree = Tree.fromstring(gold_tree)
                    gold_tree.set_label('Gold')
                    lines += '\n' + '\n'.join(draw_str_lines(gold_tree))
                    lines += '\n\n' + pred_tree + '\n\n\n'
                fw.write(lines)

        return continuous_score_desc_logg(scores) + (None,)