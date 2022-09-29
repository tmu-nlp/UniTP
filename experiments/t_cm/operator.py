import torch
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, tune_epoch_type, frac_06
from models.utils import fraction, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper.co import CO, CVP
from experiments.helper import make_tensors, speed_logg, continuous_score_desc_logg
from data.penn_types import C_PTB


train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     chunk = BaseType(0.5, validator = frac_open_0)),
                  chunk_hinge_loss = true_type,
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))
                #   keep_low_attention_rate = BaseType(1.0, validator = frac_close),

def serialize_matrix(m, skip = None):
    for rid, row in enumerate(m):
        offset = 0 if skip is None else (rid + skip)
        for cid, val in enumerate(row[offset:]):
            yield rid, offset + cid, float(val)

def sort_matrix(m, lhs, rhs, higher_better):
    lines = []
    n = max(len(n) for n in lhs) + 1
    for rid, row in enumerate(m):
        line = []
        for cid, _ in sorted(enumerate(row), key = lambda x: x[1], reverse = higher_better):
            line.append(rhs[cid])
        lines.append(lhs[rid].ljust(n) + ': ' + ' '.join(line))
    return '\n'.join(lines)

def save_txt(fname, append, lhv, rhv, dst, cos):
    with open(fname, ('w', 'a+')[append]) as fw:
        if append:
            fw.write('\n\n')
            lhv, rhv = lhv.label, rhv.label
            n = 'Label'
        else:
            lhv, rhv = lhv.tag, rhv.tag
            n = 'Tag'
        fw.write(f'Distance\n  {n}:\n')
        fw.write(sort_matrix(dst, lhv, rhv, False))
        fw.write(f'\nCosine\n  {n}:\n')
        fw.write(sort_matrix(cos, lhv, rhv, True))

class CMOperator(CO):
    def _step(self, mode, ds_name, batch, batch_id = None):

        batch['key'] = corp = ds_name if self.multi_corp else None
        if mode == M_TRAIN:
            batch['supervision'] = gold_chunks = batch['chunk'][:, :-2] # top 2 are stable ones and useless
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

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            tag_mis      = (tags    != batch['tag'  ])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | existences[:, tag_start: tag_end])
            label_weight = (label_mis | existences[:, tag_start:])
            
            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, label_logits, label_weight, corp)
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
            if serial: # [tree, length, token, tag, label, chunk, b_seg, segment, weight, vote]
                if draw_weights:
                    b_data.append(mean_stdev(weight).type(torch.float16))
                    b_data.append(None if chunk_vote is None else chunk_vote.type(torch.float16))
                else:
                    b_data += [None, None]
            vis.process(batch_id, make_tensors(*b_head, *b_data))
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_bins, test_bins = self._mode_length_bins
        devel_init, test_init = self._initial_run
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
            if self.multi_corp:
                flush_heads = test_init[ds_name]
                test_init[ds_name] = False
            else:
                flush_heads = test_init
                self._initial_run = devel_init, False
            length_bins = test_bins
        else:
            folder = ds_name + '_devel'
            if int(epoch_minor) == 0:
                draw_weights = is_bin_times(int(epoch_major))
            else:
                draw_weights = False
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
        work_dir = self.recorder.create_join(folder)
        
        if serial := (draw_weights or flush_heads or self.dm is None):
            vis = CMVA(epoch,
                       work_dir,
                       self._evalb,
                       i2vs,
                       self.recorder.log,
                       ds_name == C_PTB,
                       draw_weights,
                       length_bins,
                       flush_heads)
        else:
            vis = CMVP(epoch, work_dir, self._evalb, self.recorder.log, self.dm, m_corp)
        vis = VisRunner(vis, async_ = serial) # wrapper
        self._vis_mode = vis, use_test_set, final_test, serial, draw_weights

        if final_test and self.multi_corp:
            work_dir = self._recorder.create_join('multilingual')
            # save vocabulary
            for corp, i2vs in self.i2vs.items():
                with open(join(work_dir, 'tag.' + corp), 'w') as fw:
                    fw.write('\n'.join(i2vs.tag))
                with open(join(work_dir, 'label.' + corp), 'w') as fw:
                    fw.write('\n'.join(i2vs.label))
            # save Tag/Label
            for prefix in ('tag', 'label'):
                if get_label := prefix == 'label':
                    fn = self._model.get_multilingual_label_matrices
                else:
                    fn = self._model.get_multilingual_tag_matrices
                for lhs, rhs, dst, cos in fn():
                    # save matrix
                    #  # 'a+' if get_label else 'w' lhv, rhv = self.i2vs[lhs], self.i2vs[rhs]
                    if lhs == rhs:
                        fname = lhs
                        save_txt(join(work_dir, lhs + '.txt'), get_label, self.i2vs[lhs], self.i2vs[lhs], dst, cos)
                    else:
                        fname = lhs + '.' + rhs
                        save_txt(join(work_dir, lhs + '.' + rhs + '.txt'), get_label, self.i2vs[lhs], self.i2vs[rhs], dst, cos)
                        save_txt(join(work_dir, rhs + '.' + lhs + '.txt'), get_label, self.i2vs[rhs], self.i2vs[lhs], dst.T, cos.T)
                    with open(join(work_dir, prefix + fname + '.csv'), 'w') as fw:
                        fw.write('type,row,col,value\n')
                        if lhs == rhs:
                            for r, c, v in serialize_matrix(dst, 1):
                                fw.write(f'd,{r},{c},{v}\n')
                            for r, c, v in serialize_matrix(cos, 1):
                                fw.write(f'c,{c},{r},{v}\n')
                        else:
                            for r, c, v in serialize_matrix(dst):
                                fw.write(f'd,{r},{c},{v}\n')
                            for r, c, v in serialize_matrix(cos):
                                fw.write(f'c,{r},{c},{v}\n')


    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial, _ = self._vis_mode
        devel_bins, test_bins = self._mode_length_bins
        if serial:
            scores, desc, logg, length_bins = vis.after()
        else:
            scores, desc, logg = vis.after()
            length_bins = None
        if self.multi_corp:
            desc = ds_name.upper() + desc
            logg = ds_name + ' ' + logg
        else:
            desc = 'Evalb' + desc

        if length_bins is not None:
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
from visualization import tee_trees
from sys import stderr
from copy import deepcopy

class CMVA(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 mark_np_without_dt,
                 draw_weights   = False,
                 length_bins    = None,
                 flush_heads    = False):
        super().__init__(epoch, work_dir, i2vs)
        self._evalb = evalb
        htree = self.join('head.tree')
        dtree = self.join(f'data.{epoch}.tree')
        self._fnames   = htree, dtree
        self._is_anew  = not isfile(htree) or flush_heads
        self._rpt_file = self.join(f'data.{epoch}.rpt')
        self._logger   = logger
        self._i2vs     = i2vs.token, i2vs.tag, i2vs.label, 
        self._length_bins = length_bins
        self._draw_file = self.join(f'data.{epoch}.art') if draw_weights else None
        self._error_idx = 0, []
        self._headedness_stat = self.join(f'data.{epoch}.headedness'), {}
        self._mark_np_without_dt = mark_np_without_dt
        for fname in listdir(work_dir):
            if 'bin_' in fname and fname.endswith('.tree'):
                if flush_heads and fname.startswith('head.') or fname.startswith('data.'):
                    remove(join(work_dir, fname))

    def _before(self):
        htree, dtree = self._fnames
        if isfile(dtree): remove(dtree)
        if self._is_anew:
            self._length_bins = set()
            if isfile(htree): remove(htree)
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