import torch
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, false_type, tune_epoch_type, frac_06, frac_close
from models.utils import fraction, mean_stdev
from models.loss import binary_cross_entropy, hinge_loss
from experiments.t_lstm_nccp.operator import PennOperator
from data.penn_types import C_PTB
from utils.shell_io import byte_style


train_type = dict(loss_weight = dict(tag   = BaseType(0.2, validator = frac_open_0),
                                     label = BaseType(0.3, validator = frac_open_0),
                                     fence = BaseType(0.5, validator = frac_open_0)),
                  fence_hinge_loss = true_type,
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))
                #   keep_low_attention_rate = BaseType(1.0, validator = frac_close),


def unpack_label_like(seq, segment):
    layers = []
    start = 0
    for size in segment:
        end = start + size
        layers.append(seq[:, start:end])
        start = end
    return layers

def unpack_fence(seq, segment, is_indices):
    layers = []
    start = 0
    for size in segment[is_indices:]:
        end = start + size + 1
        layers.append(seq[:, start:end])
        start = end
    return layers

def extend_fence_idx(unpacked_fence_idx):
    layers = []
    first = unpacked_fence_idx[0]
    bs = first.shape[0]
    batch_dim = torch.arange(bs, device = first.device)[:, None]
    for layer in unpacked_fence_idx:
        full_layer = torch.zeros(bs, layer.max() + 1, dtype = torch.bool, device = first.device)
        full_layer[batch_dim, layer] = True
        layers.append(full_layer)
    return torch.cat(layers, dim = 1)

def unpack_fence_vote(fence_vote, batch_size, segment):
    layers = []
    start = 0
    for size in segment:
        end = start + (size + 1) * size
        layers.append(fence_vote[:, start:end].reshape(batch_size, size + 1, size))
        start = end
    return layers

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

class MultiOperator(PennOperator):
    def __init__(self, model, get_datasets, recorder, i2vs, get_dm, evalb, train_config):
        super().__init__(model, get_datasets, recorder, i2vs, get_dm, evalb, train_config)

    def _step(self, mode, ds_name, batch, batch_id = None):

        supervised_signals = {}
        supervised_signals['key'] = corp = ds_name if self.multi_corp else None
        if mode == M_TRAIN:
            supervised_signals['supervised_fence'] = gold_fences = unpack_fence(batch['fence'], batch['segment'], True)
            # supervised_signals['keep_low_attention_rate'] = self._train_config.keep_low_attention_rate
        if 'sub_idx' in batch:
            supervised_signals['sub_idx'] = batch['sub_idx']
        if 'sub_fence' in batch:
            supervised_signals['sub_fence'] = batch['sub_fence']
        elif 'plm_idx' in batch:
            for x in ('plm_idx', 'plm_start'):
                supervised_signals[x] = batch[x]

        batch_time = time() # weight, fence, fence_idx, fence_vote, segment, seg_length
        (batch_size, batch_len, static, top3_label_logits,
         embeddings, existences, _, tag_logits, label_logits,
         (weights, fence_logits, fence_idx, fence_vote,
          segment, seg_length)) = self._model(batch['token'], self._tune_pre_trained, **supervised_signals)
        batch_time = time() - batch_time

        fences = fence_logits > 0
        if not self._train_config.fence_hinge_loss:
            fence_logits = self._sigmoid(fence_logits)

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, :batch_len]
            tag_mis      = (tags    != batch['tag'])
            label_mis    = (labels  != batch['label'])
            tag_weight   = (  tag_mis | bottom_existence)
            label_weight = (label_mis | existences)
            extended_gold_fences = extend_fence_idx(gold_fences)
            
            tag_loss, label_loss = self._model.get_losses(batch, None, tag_logits, label_logits, corp)
            if self._train_config.fence_hinge_loss:
                fence_loss = hinge_loss(fence_logits, extended_gold_fences, None)
            else:
                fence_loss = binary_cross_entropy(fence_logits, extended_gold_fences, None)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.fence * fence_loss + total_loss
            total_loss.backward()
            
            if self.recorder._writer is not None:
                suffix = ds_name if self.multi_corp else None
                if hasattr(self._model, 'tensorboard'):
                    self._model.tensorboard(self.recorder, self.global_step)
                self.recorder.tensorboard(self.global_step, 'Accuracy/%s', suffix,
                    Tag   = 1 - fraction(tag_mis,     tag_weight),
                    Label = 1 - fraction(label_mis, label_weight),
                    Fence = fraction(fences == extended_gold_fences))
                self.recorder.tensorboard(self.global_step, 'Loss/%s', suffix,
                    Tag   = tag_loss,
                    Label = label_loss,
                    Fence = fence_loss,
                    Total = total_loss)
                batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
                if 'segment' in batch:
                    batch_kwargs['Height'] = batch['segment'].shape[0]
                self.recorder.tensorboard(self.global_step, 'Batch/%s', suffix, **batch_kwargs)
        else:
            vis, _, _, serial, draw_weights = self._vis_mode

            tags   = self._model.get_decision(tag_logits  ).type(torch.uint8).cpu().numpy()
            labels = self._model.get_decision(label_logits).type(torch.uint8).cpu().numpy()
            fences = fence_idx.type(torch.int16).cpu().numpy()
            seg_length = seg_length.type(torch.int16).cpu().numpy()
            if fence_vote is not None and draw_weights:
                fence_vote = fence_vote.type(torch.float16).cpu().numpy()
            if serial:
                b_head = tuple((batch[x].type(torch.uint8) if x in ('tag', 'label', 'fence') else batch[x]).cpu().numpy() for x in 'token tag label fence'.split())
                b_head = b_head + (batch['segment'].cpu().numpy(), batch['seg_length'].cpu().numpy())
                # length, token, tag, label, fence, segment, seg_length

                weight = mean_stdev(weights).cpu().numpy() if draw_weights else None
                b_data = (tags, labels, fences, fence_vote, weight, segment, seg_length)
            else:
                b_head = (batch_id, batch['token'].cpu().numpy())
                # segment, token, tag, label, fence, seg_length
                b_data = (tags, labels, fences, segment, seg_length)

            vis.process(b_head + b_data)
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
        serial = draw_weights or flush_heads or self.dm is None
        if serial:
            async_ = True
            vis = MultiVis(epoch,
                          work_dir,
                          self._evalb,
                          i2vs,
                          self.recorder.log,
                          ds_name == C_PTB,
                          draw_weights,
                          length_bins,
                          flush_heads)
        else:
            async_ = False
            vis = ParallelVis(epoch, work_dir, self._evalb, self.recorder.log, self.dm, m_corp)
        vis = VisRunner(vis, async_ = async_) # wrapper
        vis.before()
        self._vis_mode = vis, use_test_set, final_test, serial, draw_weights

        if final_test and hasattr(self._model, 'get_multilingual'):
            work_dir = self._recorder.create_join('multilingual')
            # save vocabulary
            for corp, i2vs in self.i2vs.items():
                with open(join(work_dir, 'tag.' + corp), 'w') as fw:
                    fw.write('\n'.join(i2vs.tag))
                with open(join(work_dir, 'label.' + corp), 'w') as fw:
                    fw.write('\n'.join(i2vs.label))
            # save Tag/Label
            for get_label in (False, True):
                prefix = ('tag', 'label')[get_label] + '.'
                for lhs, rhs, dst, cos in self._model.get_multilingual(get_label):
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
        desc = (ds_name.upper() if self.multi_corp else 'Evalb') + desc

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

        speed_outer = float(f'{count / seconds:.1f}')
        speed_inner = float(f'{count / vis.proc_time:.1f}') # unfolded with multiprocessing
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)

        if serial:
            dmt = speed_dm = ''
        else:
            dmt = self._dm.duration
            speed_dm = f' ◇ {count / dmt:.1f}'
            dmt = f' ◇ {dmt:.3f}'
            desc += byte_style(speed_dm + 'sps.', '2')

        logg += f' @{speed_outer} ◇ {speed_inner}{speed_dm} sps. (sym:nn {rate:.2f}; {seconds:.3f}{dmt} sec.)'
        scores['speed'] = speed_outer
        if not final_test:
            prefix = 'TestSet' if use_test_set else 'DevelSet'
            suffix = ds_name if self.multi_corp else None
            self.recorder.tensorboard(self.global_step, prefix + '/%s', suffix,
                                      F1 = scores.get('F1', 0),
                                      SamplePerSec = None if serial else speed_dm)
        return scores, desc, logg

    def _get_optuna_fn(self, train_params):
        assert self.multi_corp
        from utils.train_ops import train, get_optuna_params
        from utils.str_ops import height_ratio

        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                data = specs['data']
                balanced = {}
                for corp, data_config in data.items():
                    balanced[corp] = data_config['balanced'] = trial.suggest_float(corp, 0.0, 1.0)
                self._train_materials = balanced, self._train_materials[1] # for train/train_initials(max_epoch>0)
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                self._train_config._nested.update(specs['train'])
                self._train_materials = balanced, self._train_materials[1] # for train/train_initials(max_epoch>0)
                return ''.join(height_ratio(b) for b in balanced.values()) + f';lr={lr:.1e}'

            self._init_mode_trees()
            self.setup_optuna_mode(spec_update_fn, trial)
            
            return train(optuna_params, self)['key']
        return obj_fn


from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove
from utils.shell_io import parseval, rpt_summary
from data.multib import get_tree_from_signals, draw_str_lines
from visualization import tee_trees
from sys import stderr

def batch_trees(b_word, b_tag, b_label, b_fence, b_segment, b_seg_length, i2vs, fb_label, b_weight = None, b_fence_vote = None, mark_np_without_dt = False):
    for sid, (word, tag, label, fence, seg_length) in enumerate(zip(b_word, b_tag, b_label, b_fence, b_seg_length)):
        layers_of_label = []
        layers_of_fence = []
        layers_of_weight = None if b_weight is None else []
        layers_of_fence_vote = None if b_fence_vote is None else []
        label_start = 0
        fence_start = 0
        fence_vote_start = 0
        for l_cnt, (l_size, l_len) in enumerate(zip(b_segment, seg_length)):
            label_layer = tuple(i2vs.label[i] for i in label[label_start: label_start + l_len])
            layers_of_label.append(label_layer)
            if l_cnt:
                layers_of_fence.append(fence[fence_start: fence_start + l_len + 1])
                fence_start += l_size + 1
            else:
                ln = l_len
            if l_len == 1:# or l_cnt > 1 and layers_of_label[-1] == layers_of_label[-2]:
                break
            if b_weight is not None:
                layers_of_weight.append(b_weight[sid, label_start: label_start + l_len])
            if b_fence_vote is not None:
                fence_vote_end = fence_vote_start + (l_size + 1) * l_size
                fence_vote_layer = b_fence_vote[sid, fence_vote_start: fence_vote_end]
                if fence_vote_layer.size: # for erroneous outputs
                    fence_vote_layer = fence_vote_layer.reshape(l_size + 1, l_size)[:l_len + 1, :l_len]
                    layers_of_fence_vote.append(fence_vote_layer)
                    fence_vote_start = fence_vote_end
            label_start += l_size
        wd = [i2vs.token[i] for i in word[:ln]]
        tg = [i2vs.tag  [i] for i in  tag[:ln]]
        yield get_tree_from_signals(wd, tg, layers_of_label, layers_of_fence, fb_label, layers_of_weight, layers_of_fence_vote, mark_np_without_dt)


class MultiVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 mark_np_without_dt,
                 draw_weights   = False,
                 length_bins    = None,
                 flush_heads    = False):
        super().__init__(epoch)
        self._evalb = evalb
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._fnames   = htree, dtree
        self._join_fn  = lambda x: join(work_dir, x)
        self._is_anew  = not isfile(htree) or flush_heads
        self._rpt_file = join(work_dir, f'data.{epoch}.rpt')
        self._logger   = logger
        self._i2vs     = i2vs
        self.register_property('length_bins',  length_bins)
        self._draw_file = join(work_dir, f'data.{epoch}.art') if draw_weights else None
        self._error_idx = 0, []
        self._headedness_stat = join(work_dir, f'data.{epoch}.headedness'), {}
        self._mark_np_without_dt = mark_np_without_dt
        for fname in listdir(work_dir):
            if 'bin_' in fname and fname.endswith('.tree'):
                if flush_heads and fname.startswith('head.') or fname.startswith('data.'):
                    remove(join(work_dir, fname))

    def _before(self):
        htree, dtree = self._fnames
        if isfile(dtree): remove(dtree)
        if self._is_anew:
            self.register_property('length_bins', set())
            if isfile(htree): remove(htree)
        if self._draw_file and isfile(self._draw_file):
            remove(self._draw_file)

    def _process(self, batch):
        (h_token, h_tag, h_label, h_fence, h_segment, h_seg_length,
         d_tag, d_label, d_fence, d_fence_vote, d_weight, d_segment, d_seg_length) = batch

        if self._is_anew:
            trees = []
            for tree in batch_trees(h_token, h_tag, h_label, h_fence, h_segment, h_seg_length, self._i2vs, None):
                trees.append(' '.join(str(tree).split()))
            self.length_bins |= tee_trees(self._join_fn, 'head', h_seg_length[:, 0], trees, None, 10)

        trees = []
        idx_cnt, error_idx = self._error_idx
        for tid, (tree, safe) in enumerate(batch_trees(h_token, d_tag, d_label, d_fence, d_segment, d_seg_length, self._i2vs, 'S')):
            idx_cnt += 1 # start from 1
            if not safe:
                error_idx.append(idx_cnt)
            trees.append(' '.join(str(tree).split()))
        self._error_idx = idx_cnt, error_idx

        if self._draw_file is None:
            bin_size = None
        else:
            bin_size = None if self.length_bins is None else 10
            _, head_stat = self._headedness_stat
            with open(self._draw_file, 'a+') as fw:
                for tree, safe, stat in batch_trees(h_token, d_tag, d_label, d_fence, d_segment, d_seg_length, self._i2vs, 'S', 
                                                    d_weight, d_fence_vote, self._mark_np_without_dt):
                    for lb, (lbc, hc) in stat.items():
                        if lb in head_stat:
                            label_cnt, head_cnts = head_stat[lb]
                            for h, c in hc.items():
                                head_cnts[h] += c
                            head_stat[lb] = lbc + label_cnt, head_cnts
                        else:
                            head_stat[lb] = lbc, hc
                    if not safe:
                        fw.write('\n[FORCING TREE WITH ROOT = S]\n')
                    try:
                        fw.write('\n'.join(draw_str_lines(tree)) + '\n\n')
                    except Exception as err:
                        print('  FAILING DRAWING:', err, file = stderr)
                        fw.write('FAILING DRAWING\n\n')
        tee_trees(self._join_fn, f'data.{self.epoch}', d_seg_length[:, 0], trees, None, bin_size)

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

        if self.length_bins is not None and self._draw_file is not None:
            with open(self._join_fn(f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self.length_bins:
                    fhead = self._join_fn(f'head.bin_{wbin}.tree')
                    fdata = self._join_fn(f'data.{self.epoch}.bin_{wbin}.tree')
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

        desc = f'({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger, self.length_bins

class ParallelVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, logger, dm, corp_key):
        super().__init__(epoch)
        self._join = lambda fname: join(work_dir, fname)
        self._fdata = self._join(f'data.{self.epoch}.tree')
        self._args = dm, evalb, logger, corp_key

    def _before(self):
        self._args[0].timeit()

    def _process(self, batch):
        batch_id, h_token, d_tag, d_label, d_fence, d_segment, d_seg_length = batch
        dm, _, _, corp_key = self._args
        dm.batch(batch_id, d_segment, h_token, d_tag, d_label, d_fence, d_seg_length, key = corp_key)
    
    def _after(self):
        dm, evalb, logger, _ = self._args
        fhead = self._join(f'head.tree')
        fdata = self._fdata

        tree_text = dm.batched()
        if tree_text: # none mean text concat without a memory travel
            with open(fdata, 'w') as fw:
                fw.write(tree_text)

        proc = parseval(evalb, fhead, fdata)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        if num_errors:
            logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'
                with open(self._join(fname), 'w') as fw:
                    fw.write(report)
                logger(f'  (Check {fname} for details.)')

        desc = f'({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger

    @property
    def save_tensors(self):
        return False

    @property
    def length_bins(self):
        return None