from torch import nn
from utils.operator import Operator
from utils.file_io import join, listdir, remove, isdir
from experiments.helper import speed_logg, WarmOptimHelper
from data.stan_types import C_SSTB

class CO(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, get_dm, evalb, train_config):
        super().__init__(model, get_datasets, recorder, i2vs, get_dm)
        self._init_mode_trees()
        self._evalb = evalb
        self._sigmoid = nn.Sigmoid()
        self._softmax = nn.Softmax(dim = 2)
        self._train_config = train_config
        self._tune_pre_trained = False

    def _init_mode_trees(self):
        if self.multi_corp:
            make = lambda v: {k: v for k in self.i2vs}
            self._mode_length_bins = make(None), make(None)
            self._initial_run      = make(True), make(True)
        else:
            self._mode_length_bins = None, None
            self._initial_run      = True, True

    def _build_optimizer(self, start_epoch):
        # self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.6 betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
        if start_epoch > 0:
            base_path = self.recorder._instance_dir[1]
            for folder in listdir(base_path):
                if folder.endswith('_devel') and isdir(fpath := join(base_path, folder)):
                    CO.clean_and_report(fpath, start_epoch)
        optim = hp.optimizer
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained.from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.tune_pre_trained.lr_factor if tune else 1
        learning_rate = self._schedule_lr(epoch, wander_ratio, lr_factor)
        self.recorder.tensorboard(self.global_step, 'Batch/%s', Learning_Rate = learning_rate, Epoch = epoch)


    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial = self._vis_mode
        if serial and ds_name != C_SSTB:
            scores, desc, logg, length_bins = vis.after()
        else:
            length_bins = None
            scores, desc, logg = vis.after()

        if self.multi_corp:
            desc = ('☺︎' if ds_name == C_SSTB else ds_name.upper()) + desc
            logg = ds_name + ' ' + logg
        else:
            desc = ('Evalb', '☺︎')[ds_name == C_SSTB] + desc

        devel_bins, test_bins = self._mode_length_bins
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
        if not final_test and self.recorder._writer is not None:
            prefix = 'TestSet' if use_test_set else 'DevelSet'
            suffix = ds_name if self.multi_corp else None
            key = dict(F1 = scores.get('F1', 0)) if ds_name != C_SSTB else scores
            self.recorder.tensorboard(self.global_step, prefix + '/%s', suffix, **key,
                                      SamplePerSec = None if serial else speed_dm)
        scores['speed'] = speed_outer
        self._vis_mode = None
        return scores, desc + _desc, logg + _logg

    def combine_scores_and_decide_key(self, epoch, ds_scores):
        key = []
        for ds_name, ds_score in ds_scores.items():
            if ds_name != C_SSTB:
                key.append(ds_score.get('F1', 0))
            else:
                key.append(ds_score['q'])
        ds_scores['key'] = sum(key) / len(key)
        return ds_scores

    @staticmethod
    def clean_and_report(fpath, start_epoch):
        removed = remove_vis_data_from(fpath, start_epoch)
        if removed:
            if len(removed) == 1:
                content = removed[0]
            else:
                content = f'{len(removed)} files'
            Operator.msg(f' [{start_epoch:.2f}:] {content} removed from {fpath}.')

        if isdir(fpath := fpath.replace('_devel', '_test_with_devel')):
            removed = remove_vis_data_from(fpath, start_epoch)
            if removed:
                if len(removed) == 1:
                    content = removed[0]
                else:
                    content = f'{len(removed)} files'
                Operator.msg(f' [{start_epoch:.2f}:] {content} removed from {fpath}.')


def remove_vis_data_from(fpath, start_epoch):
    removed = []
    for fname in listdir(fpath):
        if fname.startswith('data.'):
            if fname.endswith('.tree'): # batch | epoch
                if '.bin_' in fname:
                    batch_or_epoch = fname[5:fname.find('.bin_')] # data.[].bin_xx.tree
                else:
                    batch_or_epoch = fname[5:-5] # data.[].tree
                if '.' in batch_or_epoch and float(batch_or_epoch) >= start_epoch:
                    remove(join(fpath, fname))
                    removed.append(fname)
            elif fname.endswith('.pkl') or fname.endswith('.rpt'): # batch_epoch
                epoch = fname[5:-4]
                if '_' in epoch:
                    epoch = epoch[epoch.index('_') + 1:]
                if float(epoch) >= start_epoch:
                    remove(join(fpath, fname))
                    removed.append(fname)
    return removed

from data.mp import BaseVis
from utils.shell_io import parseval, rpt_summary
from experiments.helper import continuous_score_desc_logg


class CVP(BaseVis):
    save_tensors = False
    
    def __init__(self, epoch, work_dir, evalb, logger, dm, corp_key):
        super().__init__(epoch, work_dir, None)
        self._fdata = self.join(f'data.{self.epoch}.tree')
        self._args = dm, evalb, logger, corp_key

    def _before(self):
        self._args[0].timeit()
    
    def _after(self):
        dm, evalb, logger, _ = self._args
        fhead = self.join(f'head.tree')
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

        if num_errors := len(errors):
            logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'
                with open(self.join(fname), 'w') as fw:
                    fw.write(report)
                logger(f'  (Check {fname} for details.)')
        return continuous_score_desc_logg(scores)