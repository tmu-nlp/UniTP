
from datetime import datetime
from utils.file_io import join, create_join, listdir, isdir, isfile, remove, rm_rf, rename
from utils.file_io import copy_with_prefix_and_rename, link, basename
from utils.yaml_io import load_yaml, save_yaml
from utils.param_ops import zip_nt_params, dict_print, fold_same_children, change_key
from utils.shell_io import byte_style
from collections import namedtuple
from sys import stderr
from itertools import count
from math import isnan
import torch

_rt_file = 'register_and_tests.yaml'
_rt_lock = 'register_and_tests.lock'
_sv_file = 'settings_and_validation.yaml'
_sv_lock = 'settings_and_validation.lock'

def _rt_file_lock(task_dir):
    rt_file = join(task_dir, _rt_file)
    rt_lock = join(task_dir, _rt_lock)
    return rt_file, rt_lock

def _sv_file_lock(instance_dir):
    sv_file = join(instance_dir, _sv_file)
    sv_lock = join(instance_dir, _sv_lock)
    return sv_file, sv_lock

dev_test_fns = ((lambda x: x.dev_score), (lambda x: x.test_score))
Trial = namedtuple('Trial', 'tid, model, dev_score, test_score, spec_string, specs')
def gen_trial_model_dev_test_string(fpath):
    trial_rt = load_yaml(*_rt_file_lock(fpath), wait = False)
    for folder in listdir(fpath):
        if not isdir(idir := join(fpath, folder)) or '.' not in folder:
            continue
        tid, trial_string = folder.split('.', 1)
        if not tid.isdecimal() or tid not in trial_rt or 'key' not in trial_rt[tid]:
            continue
        dev_specs = load_yaml(*_sv_file_lock(idir))
        dev_model, dev_score = max(dev_specs['results'].items(), key = lambda x: x[1])
        yield Trial(tid, dev_model, dev_score, trial_rt[tid]['key'], trial_string, dev_specs)

def summary_trials(fpath, fname):
    trials = sorted(gen_trial_model_dev_test_string(fpath),
                    key = dev_test_fns[0], reverse = True)
    rank_n = max(len(t.tid) for t in trials) + 1
    best_by_dev = trials[0]
    def write(fw, trials):
        for eid, t in enumerate(trials):
            fw.write(f'{eid:{rank_n}d}.[{t.test_score:.2f} {t.dev_score:.2f}]  ')
            fw.write(f'#{t.tid}'.rjust(rank_n) + '.' + t.model + '  ')
            fw.write(t.spec_string + '\n')

    with open(join(fpath, fname), 'w') as fw:
        fw.write('Rank [test *dev]  #Trail.best_model  Hyper-parameters\n')
        write(fw, trials)
        fw.write('\n\n')
        fw.write('Rank [*test dev]  #Trail.best_model  Hyper-parameters\n')
        trials.sort(key = dev_test_fns[1], reverse = True)
        write(fw, trials)
    return best_by_dev, trials[0]


class Recorder:
    '''A Recorder provides environment for an Operator, created in a Manager, operated by the Operator.'''
    
    def __init__(self, task_dir, get_configs, config_dict_or_instance, instance_name = None, keep_top_k = 4, evalb = None, child_mode = False):
        new_instance = isinstance(config_dict_or_instance, dict)

        rt_file, rt_lock = _rt_file_lock(task_dir)
        if new_instance:
            rt, unlock = load_yaml(rt_file, rt_lock, wait_then_block = True, wait_or_exit = not child_mode)
            if len(rt):
                name_len = max(len(i) for i in rt.keys())
                existing = set(int(i) for i in rt.keys())
                for instance in count():
                    if instance in existing:
                        continue
                    break
            else:
                name_len = 2
                instance = 0
            
            if len(instance := str(instance)) < name_len:
                instance = '0' * (name_len - len(instance)) + instance
            rt[instance] = {}
            unlock()
            save_yaml(rt, rt_file, rt_lock) # final confirm
            if instance_name:
                instance_dir = f'{instance}.{instance_name}'
            else:
                instance_dir = instance
            instance_dir = create_join(task_dir, instance_dir)
            if 'results' not in config_dict_or_instance:
                config_dict_or_instance['results'] = {}
            sv_file, sv_lock = _sv_file_lock(instance_dir)
            save_yaml(config_dict_or_instance, sv_file, sv_lock)
        else:
            rt = load_yaml(rt_file, rt_lock)
            for instance_dir in listdir(task_dir):
                instance = instance_dir.split('.')[0]
                if instance.isdigit() and int(instance) == int(config_dict_or_instance):
                    break
                instance = None
            assert instance in rt, f'instance {config_dict_or_instance} not registered.'
            instance_dir = create_join(task_dir, instance_dir)
            sv_file, sv_lock = _sv_file_lock(instance_dir)
            assert isfile(sv_file), f"'{sv_file}' is not found."

        self._instance_dir = instance, instance_dir
        self._test_metrics = rt[instance]
        self._get_configs  = get_configs
        self._ckpt_fname = join(instance_dir, 'checkpoint')
        self._model_dir  = create_join(instance_dir, 'models')
        _, self._sv_unlock = load_yaml(sv_file, sv_lock, wait_then_block = True)
        self._rt_file_lock = rt_file, rt_lock
        self._sv_file_lock = sv_file, sv_lock
        self._writer = None
        self._keep_top_k = keep_top_k
        self._evalb = evalb
        self.log(datetime.now())

    def new_trial_recorder(self, specs_update_fn, trial, fpath):
        assert self._test_metrics, 'Current version does not allow direct optuna'
        specs = load_yaml(*self._sv_file_lock, wait = False)
        dev_results    = specs.pop('results')
        best_model     = max(dev_results, key = lambda x: dev_results[x]);
        best_score     = dev_results.pop(best_model)
        specs['results'] = {best_model: best_score}
        trial_name     = specs_update_fn(specs, trial)
        child_recorder = Recorder(fpath, self._get_configs, specs, trial_name, 3, self._evalb, True)
        link(join(self._model_dir, best_model), join(child_recorder._model_dir, best_model))
        _, child_dir   = child_recorder._instance_dir
        child_dir = basename(child_dir)
        self.msg('» ' + byte_style(child_dir, '2'))
        trial.set_user_attr('dir', child_dir)
        self.log(f'⌙→ Trial [{child_dir}] on best model {best_model}')
        return child_recorder
        
    def best_trial(self, fpath = 'trials', fname = 'rank.txt', by_test_score = False):
        assert callable(self._sv_unlock), 'Not main recorder?'
        if isdir(fpath := join(self._instance_dir[1], fpath)):
            return summary_trials(fpath, fname)[by_test_score]

    def detach(self):
        if callable(self._sv_unlock):
            self._sv_unlock()
        _, instance_dir = self._instance_dir
        self.msg('Close ' + byte_style(instance_dir, '3'))

    def delete_all(self):
        instance, instance_dir = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt.pop(instance)
        save_yaml(rt, *self._rt_file_lock)
        rm_rf(instance_dir, stderr)

    def delete_most(self):
        _, instance_dir = self._instance_dir
        remove(join(instance_dir, 'checkpoint'))
        with open(join(instance_dir, 'experiment.log'), 'a+') as fw:
            for fname in listdir(instance_dir):
                fpath = join(instance_dir, fname)
                if isdir(fpath):
                    rm_rf(fpath, fw)

    def log(self, *args, **kwargs):
        _, instance_dir = self._instance_dir
        with open(join(instance_dir, 'experiment.log'), 'a+') as fw:
            kwargs['flush'] = True
            kwargs['file']  = fw
            print(*args, **kwargs)

    def init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            Recorder.msg(byte_style('(tensorboard is not installed; not tracking training statistics)', '3'))
            SummaryWriter = None
        if SummaryWriter is not None:
            fpath = self.create_join('train')
            self._writer = SummaryWriter(fpath)
            Recorder.msg(byte_style('Tracking training statistics:', '2'), fpath)

    def tensorboard(self, step, prefix, suffix = None, **kwargs):
        if self._writer is None:
            return
        for key, value in kwargs.items():
            if value is None: continue
            key = prefix % key
            if suffix:
                key = key + '/' + suffix
            self._writer.add_scalar(key, value, step)

    def tensorboard_histogram(self, step, key, vector):
        if self._writer is None:
            return
        self._writer.add_histogram(key, vector, step)

    @staticmethod
    def msg(*args, **kwargs):
        print(*args, **kwargs, file = stderr, flush = True)

    def task_specs(self, ignore_missing_keys = False):
        from utils.param_ops import HParams
        specs = load_yaml(*self._sv_file_lock, wait = False)
        _, model_type, train_type = self._get_configs()
        model_config = get_obj_from_config(model_type, specs['model'], ignore_missing_keys)
        train_config = get_obj_from_config(train_type, specs['train'], ignore_missing_keys)
        train_config = HParams(train_config)
        return specs['data'], model_config, train_config, specs['results']

    def create_join(self, *args):
        _, instance_dir = self._instance_dir
        return create_join(instance_dir, *args)

    @property
    def validated_models(self):
        if isdir(self._model_dir):
            models  = listdir(self._model_dir)
            results = load_yaml(*self._sv_file_lock, wait = False)['results']
            results = [(k,v) for k,v in results.items() if k in models]
            if results:
                results.sort(key = lambda x: x[1], reverse = True)
                return results

    def initial_or_restore(self, model, restore_nth_best_validated_model = None):
        model_fname = restore_opt_fn = None
        if restore_nth_best_validated_model is None and isfile(self._ckpt_fname):
            # if not set_vocab(vis_path, r_pu_su[0].py_vocabs, vocab_size):
            # recorder.set_resume_cleaner(lambda mj, mn: clean_epoch(vis_path, mj)) # no mn
            # self._path = vis_path
            # self._init = None
            # # self._pool = []

            # def list_func(self, *token):
            # if self._init is None or self._init == token:
            # if self._init is None:
            # clean_tree_heads(self._path)
            model_fname = self._ckpt_fname
        
        elif models := self.validated_models:
            nth = restore_nth_best_validated_model or 0
            model_fname = join(self._model_dir, models[nth][0])

        if model_fname is None:
            epoch = global_step = 0
            fine_validation = False
            md = dict(model.named_parameters())
            total = 0
            for t in md.values():
                x = 1
                for s in t.shape:
                    x *= s
                total += x
            md = ((k, '*'.join(str(s) for s in v.shape)) for k, v in md.items())
            self.log(dict_print(fold_same_children(zip_nt_params(md))))
            self.log('Total:', total)
        else:
            checkpoint = torch.load(model_fname)
            model_state = checkpoint['model_state_dict']
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict = False)
            if not all(x.startswith('_input_emb.') and x.endswith('._main_emb_layer.weight') for x in missing_keys) or unexpected_keys:
                for mk in missing_keys:
                    mks = set(mk.split('.'))
                    if unexpected_keys:
                        uk_scores = {}
                        for uk in unexpected_keys:
                            uks = set(uk.split('.'))
                            uk_scores[uk] = len(mks & uks)
                        uk_list = sorted(uk_scores, key = uk_scores.get, reverse = True) # by shape
                        prompt = 'Map ' + byte_style(mk, '2') + ' to:\n'
                        prompt += byte_style('-1: delete\n', '1')
                        prompt += '\n'.join(f' {eid}: ' + byte_style(uk, '3') for eid, uk in enumerate(uk_list))
                        prompt += '\n Your choice: '
                        choice = None
                        while not choice or not (choice == '-1' or choice.isdigit() and (choice := int(choice)) < len(uk_list)):
                            choice = input(prompt).strip()
                            if choice == 'q':
                                print('Exit. Please try again.', file = stderr)
                                exit()
                        if choice == '-1':
                            model_state.pop(mk)
                        else:
                            uk = unexpected_keys.pop(choice)
                            change_key(model_state, uk, mk)
                model.load_state_dict(model_state, strict = False)
                decision = input(f'Save change to {model_fname}? [Y]')
                if decision == 'Y':
                    torch.save(checkpoint, model_fname)

            if 'optimizer_state_dict' in checkpoint:
                def restore_opt_fn(optimizer):
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except ValueError as e:
                        self.msg(byte_style('Optimzer loading error:' , '3'), e)
            epoch, fine_validation, global_step = checkpoint['status']
            self._test_metrics['key'] = key = checkpoint['key']
            
            self.log('Model restored from', model_fname, f'dev[{key:.2f}]')
            Recorder.msg('Restore model (' + byte_style(f'{epoch:.2f}', '3') + ') with dev score ' + byte_style(f'{key:.2f}', '3'))
            if isinstance(restore_nth_best_validated_model, int):
                return epoch, global_step
            epoch = int(epoch)
        return epoch, fine_validation, global_step, restore_opt_fn

    def check_betterment(self, epoch, falling, global_step, model, optimizer, key):
        if isnan(key):
            key = float('-inf')
        specs = load_yaml(*self._sv_file_lock, wait = False)
        old_key = self._test_metrics.get('key')
        betterment = (old_key is None or old_key < key)
        in_top_k = any(k < key for k in specs['results'].values())
        fine_validation = falling and not betterment
        torch.save({'model_state_dict':         model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'status': (epoch, fine_validation, global_step),
                    'key': key}, self._ckpt_fname)
        if betterment or in_top_k:
            if betterment:
                self._test_metrics['key'] = key
            model_fname = timestamp(epoch)
            copy_with_prefix_and_rename(self._ckpt_fname, self._model_dir, model_fname)
            results = specs['results'], listdir(self._model_dir)
            results = {k: v for k, v in results[0].items() if k in results[1]}
            results[model_fname] = key; specs['results'] = results
            if len(results) > self._keep_top_k:
                weakest_model = min(results, key = results.get)
                self.log(' Replace worst model', weakest_model, 'with a',
                    'new best' if betterment else 'better', 'model', 
                    model_fname, f'(+{key - results.pop(weakest_model)}).')
                if isfile(weakest_model := join(self._model_dir, weakest_model)):
                    remove(weakest_model)
            else:
                self.log(' A new', 'best' if betterment else 'better', 'model', model_fname)
            save_yaml(specs, *self._sv_file_lock, wait_lock = False)
        else:
            self.log()
        return betterment

    def register_test_scores(self, scores):
        instance, _ = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt[instance] = dict(scores)
        try:
            save_yaml(rt, *self._rt_file_lock)
        except Exception as e:
            rt.pop(scores)
            save_yaml(rt, *self._rt_file_lock)
            raise e

    @staticmethod
    def experiments_status(task_path):
        rt_file = join(task_path, _rt_file)
        rt_lock = join(task_path, _rt_lock)
        (instance_status, unlock), modifed = load_yaml(rt_file, rt_lock, wait_then_block = True), False
        status = dict(locking = [], unlocked = [], other = [], tested = [])
        folders = listdir(task_path)

        name_len = 0
        instance_folders = []
        for fx in folders:
            if '.' in fx:
                sep = fx.index('.')
                instance = fx[:sep]
                exp_name = fx[sep+1:]
            else:
                instance = fx
                exp_name = None
            instance_path = join(task_path, fx)
            if isdir(instance_path):
                if instance in instance_status:
                    name_len = max(name_len, len(instance))
                    if isfile(join(instance_path, _sv_lock)):
                        status['locking'].append(instance_path) # avoid ongoing experiments
                    else:
                        instance_folders.append((instance, exp_name, fx, instance_path))
                else:
                    status['other'].append(instance_path)

        rename_list = []
        instance_folders.sort(key = lambda x: int(x[0]))
        for _cnt, (instance, exp_name, folder, fpath) in enumerate(instance_folders):
            _instance = str(_cnt)
            ap_zeros  = name_len - len(_instance)
            _instance = '0' * ap_zeros + _instance
            modify = instance != _instance
            if modify:
                new_folder = f'{_instance}.{exp_name}' if exp_name else _instance
                new_fpath = join(task_path, new_folder)
                change_key(instance_status, instance, _instance)
                rename_list.append((fpath, new_fpath))
                fpath = new_fpath + '\t<- ' + folder
                instance = _instance
                modifed = True
            key = instance_status[instance].get('key')
            if key:
                status['tested'].append(f'({key:.2f})    {fpath}')
            else:
                status['unlocked'].append(f'(?)            {fpath}')

        unlock()
        if modifed:
            save_yaml(instance_status, rt_file, rt_lock)
            for fpath, new_fpath in rename_list:
                rename(fpath, new_fpath)
        return status

    @property
    def evalb(self):
        return self._evalb

    @property
    def key_score(self):
        return self._test_metrics['key']

    @property
    def test_metrics(self):
        return self._test_metrics

from utils.param_ops import zip_nt_params, iter_zipped_nt_params
def get_obj_from_config(types, configs, ignore_missing_keys = False):
    model_params = {}
    for k, vi, vj in iter_zipped_nt_params(types, configs, ignore_missing_keys):
        model_params[k] = vi[vj]
    return zip_nt_params(model_params)

def timestamp(main, prefix = 'M'):
    if isinstance(main, str):
        return float(main[1:])
    return f'{prefix}{main:.2f}'