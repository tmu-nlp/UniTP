from utils.types import M_TRAIN, M_DEVEL, M_TEST
from numpy.random import choice
from tqdm import tqdm
from time import time
from datetime import timedelta
from torch import nn, no_grad
from utils.recorder import Recorder, timestamp
from utils.emoji import get_train_validation_pair
import torch

class Operator:
    '''An (abstract) Operator operate a customized nn.Module for training, validation and testing.
    To operator, it feeds the model with multi-tasking batch from the customized get_datasets function,
    uses the environmental Recorder to record the results of the model, and i2vs to help a Vis to visualize them.'''
    def __init__(self, model, get_datasets, recorder, i2vs):
        assert isinstance(model, nn.Module)
        assert callable(get_datasets)
        assert isinstance(recorder, Recorder)
        assert 'token' in i2vs._nested
        self._model = model
        self._get_datasets = get_datasets
        self._recorder = recorder
        self._i2vs = i2vs
        self._optimizer = None
        self._train_materials = None
        self._validate_materials = None
        self._test_materials = (_, ds_names, _) = self.get_materials(M_TEST)
        self._ds_icons = {ds_name: icon for ds_name, icon in zip(ds_names, '⚀⚁⚂⚃⚄⚅')}

    def get_materials(self, mode):
        self._epoch_start = time()
        ds_specs = self._get_datasets(mode)
        ds_specs = ((dn,) + ds for dn, ds in ds_specs.items())
        ds_names, ds_freqs, ds_iters = zip(*ds_specs)
        ds_total = sum(ds_freqs)
        return ds_total, ds_names, ds_iters

    def train_initials(self):
        assert self._train_materials is None
        train_icon, devel_icon = get_train_validation_pair()
        self._train_materials = self._get_datasets(M_TRAIN), train_icon
        self._validate_materials = self.get_materials(M_DEVEL), devel_icon
        (epoch, fine_validation, global_step) = self._recorder.initial_or_restore(self._model)
        self._optimizer = self._build_optimizer(epoch)
        self._global_step = global_step
        return epoch, fine_validation

    def train_step(self, epoch_cnt, wander_ratio, update_every_n_batch = 1):
        ds_specs, train_icon = self._train_materials
        ds_freqs = {dn: ds.size       for dn, ds in ds_specs.items()}
        ds_iters = {dn: iter(ds.iter) for dn, ds in ds_specs.items()}
        with tqdm(total = sum(ds_freqs.values()), desc = train_icon) as qbar:
            while sum(ds_freqs.values()):
                # prepare datasets for joint tasks
                total = sum(ds_freqs.values())
                ds_names, ds_probs = zip(*((dn, df/total) for dn, df in ds_freqs.items()))
                ds_name = choice(ds_names, p = ds_probs)
                ds_icon = self._ds_icons[ds_name] if len(ds_names) > 1 else ''
                batch = next(ds_iters[ds_name])

                self._schedule(epoch_cnt + qbar.n / qbar.total, wander_ratio)
                with torch.autograd.set_detect_anomaly(True):
                    num_samples, seq_len = self._step(M_TRAIN, ds_name, batch) # neural core
                if self._global_step % update_every_n_batch == update_every_n_batch - 1:
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                # display
                qbar.update(num_samples)
                qbar.desc = f'[{epoch_cnt}] {train_icon}{100*wander_ratio:.0f}% {ds_icon}:{num_samples}×{seq_len}'
                ds_freqs[ds_name] -= num_samples
                self._global_step += 1

                updated_wander_ratio = yield qbar.n / qbar.total
                if updated_wander_ratio is not None:
                    wander_ratio = updated_wander_ratio
        # next epoch

    def validate_betterment(self, epoch, falling):
        (ds_total, ds_names, ds_iters), devel_icon = self._validate_materials
        scores, ds_logg, from_start = self.validation_or_test(M_DEVEL, ds_total, ds_names, ds_iters, devel_icon, epoch)
        self._recorder.log(timestamp(epoch, 'Validation ') + f' - {ds_logg} ({from_start} from start)', end = '.')
        return self._recorder.check_betterment(epoch, falling, self._global_step, self._model, self._optimizer, scores['key'])

    def test_model(self, epoch = None):
        ds_total, ds_names, ds_iters = self._test_materials
        final_test = epoch is None
        if final_test:
            prefix = 'Test ' # final label
            epoch, self._global_step = self._recorder.initial_or_restore(self._model, restore_from_best_validation = True)
        else:
            prefix = '   ⌙→ Test ' # match length of validation
        scores, ds_logg, from_start = self.validation_or_test(M_TEST, ds_total, ds_names, ds_iters, '🔮', epoch, final_test)
        self._recorder.log(timestamp(epoch, prefix) + f' - {ds_logg} ({from_start} from start).')
        return scores

    def validation_or_test(self, mode, ds_total, ds_names, ds_iters, icon, epoch, final_test = False):
        ds_desc   = []
        ds_logg   = []
        ds_scores = {}
        self._model.eval() # stack
        epoch_stamp = timestamp(epoch, '')
        count = 0
        with tqdm(total = ds_total, desc = f'#{icon}{epoch_stamp}') as qbar:
            for ds_name, ds_iter in zip(ds_names, ds_iters):
                self._before_validation(ds_name, f'{epoch:08.2f}', mode == M_TEST, final_test)
                start, cnt = time(), 0
                for batch_id, batch in enumerate(ds_iter):
                    with no_grad():
                        num_samples, _ = self._step(mode, ds_name, batch, batch_id = batch_id)
                    cnt += num_samples
                    qbar.update(num_samples)
                scores, desc, logg = self._after_validation(ds_name, cnt, time() - start) # evalb time is excluded
                ds_desc  .append(desc)
                ds_logg  .append(logg)
                ds_scores[ds_name] = scores
            qbar.total = None
            from_start = timedelta(seconds = int(time() - self._epoch_start))
            qbar.desc = f'[{epoch_stamp}] {icon} {from_start} ' + ' '.join(ds_desc)
            ds_logg = '\n'.join(ds_logg)
        self._model.train() # restore
        scores = self.combine_scores_and_decide_key(epoch, ds_scores)
        return scores, ds_logg, from_start

    def _schedule(self, epoch, wander_ratio):
        pass

    def _step(self, mode, ds_name, batch, flush = True, batch_id = None):
        raise NotImplementedError()

    def _build_optimizer(self, start_epoch):
        raise NotImplementedError()

    def _before_validation(self, ds_name, epoch, use_test_set, final_test):
        raise NotImplementedError()

    def _after_validation(self, ds_name, count, seconds):
        raise NotImplementedError()

    @property
    def recorder(self):
        return self._recorder

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def i2vs(self):
        return self._i2vs

    @property
    def global_step(self):
        return self._global_step
    
    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        raise NotImplementedError('Should manually decide the key')

    @staticmethod
    def msg(*args, **kw_args):
        Recorder.msg(*args, **kw_args)


class CsvWriter:
    def __init__(self, fpath):
        self._file_headers = open(fpath, 'a+'), None

    def write(self, outputs):
        fw, headers = self._file_headers
        if headers is None:
            headers = tuple(outputs.keys())
            self._file_header = fw, headers
            fw.write(','.join(headers) + '\n')
        fw.write(','.join(str(outputs[h]) for h in headers) + '\n')