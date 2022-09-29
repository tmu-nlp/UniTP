from utils.operator import Operator
from data.cross.evalb_lcfrs import read_param
from utils.shell_io import has_discodop, byte_style
from utils.param_ops import get_sole_key
from experiments.helper import speed_logg, WarmOptimHelper

class DO(Operator):
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
        self._train_config = train_config
        self._tune_pre_trained = False
        self._evalb_lcfrs_kwargs = read_param(evalb_lcfrs_prm)
        self._init_mode_trees()

    def _init_mode_trees(self):
        if self.multi_corp:
            make = lambda v: {k: v for k in self.i2vs}
            self._mode_trees = make([]), make([])
        else:
            self._mode_trees = [], []

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

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial = self._vis_mode
        if serial:
            scores, desc, logg, heads = vis.after()
            devel_head, test_head = self._mode_trees
            if self.multi_corp:
                if use_test_set:
                    test_head [ds_name] = heads
                else:
                    devel_head[ds_name] = heads
            else:
                if use_test_set:
                    self._mode_trees = devel_head, heads
                else:
                    self._mode_trees = heads, test_head
        else:
            scores, desc, logg = vis.after()

        _desc, _logg, speed_outer, speed_dm = speed_logg(count, seconds, None if serial else self._dm)
        if not final_test:
            prefix = 'TestSet' if use_test_set else 'DevelSet'
            suffix = ds_name if self.multi_corp else None
            self.recorder.tensorboard(self.global_step, prefix + '/%s', suffix,
                                      F1 = scores.get('TF', 0), DF = scores.get('DF', 0),
                                      SamplePerSec = None if serial else speed_dm)
        scores['speed'] = speed_outer
        self._vis_mode = None
        return scores, desc + _desc, logg + _logg

    def combine_scores_and_decide_key(self, epoch, ds_scores):
        if self.multi_corp:
            key = [s.get('TF', 0) for s in ds_scores.values()]
            ds_scores['key'] = sum(key) / len(key)
            return ds_scores
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores.get('TF', 0.0) #f_score(scores.get('TF', 0.0), scores.get('DF', 0.0))
        return scores