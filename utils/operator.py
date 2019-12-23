from utils.types import M_TRAIN, M_DEVEL, M_TEST
from numpy.random import choice
from tqdm import tqdm
from time import time
from torch import nn, no_grad
from utils.recorder import Recorder, timestamp
class Operator:
    '''An (abstract) Operator operate a customized nn.Module for training, validation and testing.
    To operator, it feeds the model with multi-tasking batch from the customized get_datasets function,
    uses the environmental Recorder to record the results of the model, and i2vs to help a Vis to visualize them.'''
    def __init__(self, model, get_datasets, recorder, i2vs):
        assert isinstance(model, nn.Module)
        assert callable(get_datasets)
        assert isinstance(recorder, Recorder)
        assert 'word' in i2vs._nested
        self._model = model
        self._get_datasets = get_datasets
        self._recorder = recorder
        self._i2vs = i2vs
        self._optimizer = None
        self._train_materials = None
        self._validate_materials = None
        self._test_materials = self.get_materials(M_TEST)

    def get_materials(self, mode):
        ds_specs = self._get_datasets(mode)
        ds_specs = ((dn,) + ds for dn, ds in ds_specs.items())
        ds_names, ds_freqs, ds_iters = zip(*ds_specs)
        ds_total = sum(ds_freqs)
        return ds_total, ds_names, ds_iters

    def train_initials(self):
        assert self._train_materials is None
        self._optimizer = self._build_optimizer()
        self._train_materials = self._get_datasets(M_TRAIN)
        self._validate_materials = self.get_materials(M_DEVEL)
        (epoch, global_step, fine_validation) = self._recorder.initial_or_restore(self._model)
        self._global_step = global_step
        return epoch, fine_validation

    def train_step(self, epoch_cnt, wander_ratio):
        ds_specs = self._train_materials
        ds_freqs = {dn: ds.size       for dn, ds in ds_specs.items()}
        ds_iters = {dn: iter(ds.iter) for dn, ds in ds_specs.items()}
        with tqdm(total = sum(ds_freqs.values())) as qbar:
            while sum(ds_freqs.values()):
                total = sum(ds_freqs.values())
                ds_names, ds_probs = zip(*((dn, df/total) for dn, df in ds_freqs.items()))
                ds_name = choice(ds_names, p = ds_probs)
                batch = next(ds_iters[ds_name])
                # with torch.autograd.set_detect_anomaly(True):
                self._schedule(epoch_cnt + qbar.n / qbar.total, wander_ratio)
                num_samples, seq_len = self._step(M_TRAIN, ds_name, batch) # neural core
                qbar.update(num_samples)
                qbar.desc = f'E-{epoch_cnt}/{100*wander_ratio:.0f}% Batch({num_samples}, {seq_len})'
                ds_freqs[ds_name] -= num_samples
                self._global_step += 1
                yield qbar.n / qbar.total

        # next epoch

    def validate_betterment(self, epoch, falling):
        ds_total, ds_names, ds_iters = self._validate_materials
        with tqdm(total = ds_total) as qbar:
            desc = ''
            for ds_name, ds_iter in zip(ds_names, ds_iters):
                vis = self._before_validation(ds_name, timestamp(epoch, ''))
                vis._before() # TODO: mpc
                start, cnt = time(), 0
                for batch_id, batch in enumerate(ds_iter):
                    with no_grad():
                        num_samples, _ = self._step(M_DEVEL, ds_names, batch, extra = (vis, batch_id))
                    cnt += num_samples
                    qbar.update(num_samples)
                    qbar.desc = f'V-{epoch:.1f}'
                speed = cnt / (time() - start)
                desc += vis._after() + f' in {speed:.2f} s/s.; '
                self._after_validation(vis)
            qbar.desc = desc[:-2] + '.'
            self._recorder.log(timestamp(epoch, 'V') + '  ' + desc[:-2])
        return self._recorder.check_betterment(epoch, falling, self._global_step, self._model, self._optimizer, self._key())

    def test_model(self, epoch = None):
        ds_total, ds_names, ds_iters = self._test_materials
        if epoch is None:
            epoch = self._recorder.initial_or_restore(self._model, restore_from_best_validation = True)
        with tqdm(total = ds_total) as qbar:
            desc = ''
            for ds_name, ds_iter in zip(ds_names, ds_iters):
                vis = self._before_validation(ds_name, timestamp(epoch, ''), use_test_set = True)
                vis._before() # TODO: mpc
                start, cnt = time(), 0
                for batch_id, batch in enumerate(ds_iter):
                    with no_grad():
                        num_samples, _ = self._step(M_TEST, ds_names, batch, extra = (vis, batch_id))
                    cnt += num_samples
                    qbar.update(num_samples)
                    qbar.desc = f'T-{epoch:.1f}'
                speed = cnt / (time() - start)
                desc += vis._after() + f' in {speed:.2f} s/s.; '
                self._after_validation(vis)
            qbar.desc = desc[:-2] + '.'
        return self._scores()

    def _schedule(self, epoch, wander_ratio):
        pass

    def _step(self, mode, ds_name, batch, flush = True, extra = None):
        raise NotImplementedError()

    def _build_optimizer(self):
        raise NotImplementedError()

    def _before_validation(self, ds_name, epoch, use_test_set = False):
        raise NotImplementedError()

    def _after_validation(self, ds_name):
        raise NotImplementedError()

    def _key(self):
        raise NotImplementedError()

    def _scores(self):
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