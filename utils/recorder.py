
from datetime import datetime
from utils.file_io import join, create_join, listdir, isdir, isfile, remove, rm_rf, rename
from utils.file_io import copy_with_prefix_and_rename, link, basename
from utils.yaml_io import load_yaml, save_yaml
from utils.param_ops import zip_nt_params, dict_print, fold_same_children, change_key
from utils.shell_io import byte_style
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

class Recorder:
    '''A Recorder provides environment for an Operator, created in a Manager, operated by the Operator.'''
    
    def __init__(self, task_dir, task_module, config_dict_or_instance, instance_name = None, keep_top_k = 4, evalb = None, child_mode = False):
        new_instance = isinstance(config_dict_or_instance, dict)

        rt_file, rt_lock = _rt_file_lock(task_dir)
        if new_instance:
            rt, unlock = load_yaml(rt_file, rt_lock, wait_then_block = True, wait_or_exit = not child_mode)
            if len(rt):
                name_len = max(len(i) for i in rt.keys())
                inames = tuple(int(i) for i in rt.keys())
                for instance in count():
                    if instance in inames:
                        continue
                    break
            else:
                instance = 0
                name_len = 1
            instance = str(instance)
            if len(instance) < name_len:
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
        self._module     = task_module
        self._ckpt_fname = join(instance_dir, 'checkpoint')
        self._model_dir  = create_join(instance_dir, 'models')
        _, self._sv_unlock = load_yaml(sv_file, sv_lock, wait_then_block = True)
        self._rt_file_lock = rt_file, rt_lock
        self._sv_file_lock = sv_file, sv_lock
        self._key = None
        self._writer = None
        self._keep_top_k = keep_top_k
        self._evalb = evalb
        self.log(datetime.now())

    def new_trial_recorder(self, specs_update_fn, trial, fpath):
        specs = load_yaml(*self._sv_file_lock, wait = False)
        specs.pop('optuna', None)
        results        = specs.pop('results')
        best_model     = max(results, key = lambda x: results[x]);
        best_score     = results.pop(best_model)
        specs['results'] = {best_model: best_score}
        trial_name     = specs_update_fn(specs, trial)
        child_recorder = Recorder(fpath, self._module, specs, trial_name, 3, self._evalb, True)
        link(join(self._model_dir, best_model), join(child_recorder._model_dir, best_model))
        _, child_dir   = child_recorder._instance_dir
        child_dir = basename(child_dir)
        self.msg('» ' + byte_style(child_dir, '2'))
        trial.set_user_attr('dir', child_dir)
        self.log(f'⌙→ Trial [{child_dir}] on best model {best_model}')
        return child_recorder

    def summary_trials(self): # should only be a super_recorder
        assert callable(self._sv_unlock), 'Not main recorder?'
        _, instance_dir = self._instance_dir

        fpath = join(instance_dir, 'trials')
        rt_file, rt_lock = _rt_file_lock(fpath)
        # some other can be empty: if v and 'key' in v
        children_rt = load_yaml(rt_file, rt_lock, wait = False)
        test_kv = ((k, v['key']) for k, v in children_rt.items() if v and 'key' in v)
        test_kv = sorted(test_kv, key = lambda x: x[1], reverse = True)[:self._keep_top_k]
        top_k = set(k for k, _ in test_kv)

        optuna_top_k = {}
        for fname in listdir(fpath):
            if not isdir(idir := join(fpath, fname)) or '.' not in fname:
                continue
            trial_id, trial_string = fname.split('.', 1)
            if not trial_id.isdecimal():
                continue
            if trial_id in top_k:
                sv_file, sv_lock = _sv_file_lock(idir)
                trial_results = load_yaml(sv_file, sv_lock)['results']
                best_validated_model, score = max(trial_results.items(), key = lambda x: x[1])
                best_trial = f'O{trial_id}-{best_validated_model}'
                link(join(fpath, fname, 'models', best_validated_model),
                     join(instance_dir, 'models', best_trial))
                optuna_top_k[best_trial] = f'{trial_string}@{score}'

        for fname in listdir(join(instance_dir, 'models')):
            if fname[0] == 'O' and isfile(fpath := join(instance_dir, 'models', fname)):
                remove(fpath)
        main_specs = load_yaml(*self._sv_file_lock, wait = False)
        main_specs['optuna'] = optuna_top_k
        save_yaml(main_specs, *self._sv_file_lock, wait_lock = False)
        return children_rt[test_kv[0][0]]
        # for fname in listdir(join(instance_dir, 'trials')):
        #     if '.' in fname:
        #         thatsit = fname.split('.')[0] == best_child
        #     else:
        #         thatsit = fname == best_child
        #     if thatsit:
        #         child_specs = load_yaml(*_sv_file_lock(join(instance_dir, 'trials', fname)))
        #         child_results = child_specs['results']
        #         best_model = max(child_results, key = lambda x: child_results[x])
        #         best_fpath = join(instance_dir, 'trials', fname, 'models', best_model)
                
        #         specs = load_yaml(*self._sv_file_lock, wait = False)
        #         results = specs['results']
        #         results[best_model] = child_results[best_model]
        #         copy_with_prefix_and_rename(best_model, self._model_dir, best_model)
                
        #         weakest_model = min(results, key = lambda x: results[x])
        #         remove(join(self._model_dir, weakest_model))
        #         results.pop(weakest_model)

        #         self.log(' Replace the worst model', weakest_model, 'with the best model from trial', best_child, best_model)
        #         save_yaml(specs, *self._sv_file_lock, wait_lock = False)
        #         return True
        # return False

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
            self._writer = SummaryWriter(self.create_join('train'))

    def tensorboard(self, step, template, **kwargs):
        if self._writer is None:
            return
        for key, value in kwargs.items():
            if value is None: continue
            self._writer.add_scalar(template % key, value, step)

    def tensorboard_histogram(self, step, key, vector):
        if self._writer is None:
            return
        self._writer.add_histogram(key, vector, step)

    @staticmethod
    def msg(*args, **kwargs):
        print(*args, **kwargs, file = stderr, flush = True)

    def task_specs(self): # TODO if not training set trainset & develset to {}
        from utils.param_ops import HParams
        specs = load_yaml(*self._sv_file_lock, wait = False)
        _, model_type, train_type = self._module.get_configs()
        model_config = get_obj_from_config(model_type, specs['model'])
        train_config = get_obj_from_config(train_type, specs['train'])
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
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model_old_dict = checkpoint['model_state_dict']
                model_new_dict = model.state_dict()
                new_keys = tuple(model_new_dict)
                for old_key in tuple(model_old_dict):
                    if old_key in new_keys:
                        continue
                    new_candidates = {}
                    old_segs = old_key.split('.')
                    old_segs.reverse()
                    for new_key in new_keys:
                        new_segs = new_key.split('.')
                        new_segs.reverse()
                        if new_segs[0] != old_segs[0]:
                            continue
                        match_depth = 0
                        for ns, os in zip(new_segs, old_segs):
                            if ns == os:
                                match_depth += 1
                        if match_depth > 0 and model_new_dict[new_key].shape == model_old_dict[old_key]:
                            new_candidates[new_key] = match_depth
                    if any(v > 1 for v in new_candidates.values()):
                        new_candidates = {k:v for k, v in new_candidates.items() if v > 1}
                    new_candidates = sorted(new_candidates, key = new_candidates.get, reverse = True)
                    if not new_candidates:
                        print(byte_style('Delete', '1') + ' ' + old_key)
                        model_old_dict.pop(old_key)
                    else:
                        if len(new_candidates) > 1:
                            prompt = f'Change {old_key} into:\n'
                            for i, k in enumerate(new_candidates):
                                prompt += f'{i}) {k}\n'
                            prompt += '-1) ' + byte_style('delete ', '1') + f'{old_key}?\n'
                            new_key = input(prompt)
                            if new_key == 'q':
                                exit()
                            new_key = int(new_key)
                            if new_key != -1:
                                assert new_key in range(len(new_candidates))
                                new_key = new_candidates[new_key]
                        else:
                            new_key = new_candidates[0]
                            more = len(new_key) - len(old_key)
                            prompt = byte_style('Rename ', '1') # red
                            if more > 0:
                                prompt += ' ' * more
                                prompt += old_key
                                prompt += byte_style('\n    as ', '2') # green
                            else:
                                more = 0 - more
                                prompt += old_key
                                prompt += byte_style('\n    as ', '2') # green
                                prompt += ' ' * more
                            prompt += new_key
                            print(prompt)
                        if isinstance(new_key, int) and new_key < 0:
                            model_old_dict.pop(old_key)
                        else:
                            change_key(model_old_dict, old_key, new_key)
                model.load_state_dict(model_old_dict)
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
            self._key = checkpoint['key']
            
            self.log('Model restored from', model_fname)
            Recorder.msg('Restore model (' + byte_style(f'{epoch:.2f}', '3') + ') with dev score ' + byte_style(f'{self._key:.2f}', '3'))
            if isinstance(restore_nth_best_validated_model, int):
                return epoch, global_step
            epoch = int(epoch)
        return epoch, fine_validation, global_step, restore_opt_fn

    def check_betterment(self, epoch, falling, global_step, model, optimizer, key):
        if isnan(key):
            key = float('-inf')
        specs = load_yaml(*self._sv_file_lock, wait = False)
        betterment = (self._key is None or self._key < key)
        in_top_k = any(old_key < key for old_key in specs['results'].values())
        fine_validation = falling and not betterment
        torch.save({'model_state_dict':         model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'status': (epoch, fine_validation, global_step),
                    'key': key}, self._ckpt_fname)
        if betterment or in_top_k:
            if betterment:
                self._key = key
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
        save_yaml(rt, *self._rt_file_lock)

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
        return self._key

from utils.param_ops import zip_nt_params, iter_zipped_nt_params
def get_obj_from_config(types, configs):
    # import pdb; pdb.set_trace()
    model_params = {}
    for k, vi, vj in iter_zipped_nt_params(types, configs):
        # if vi.is_valid(vj):
        #     model_params[k] = vj
        # else:
        model_params[k] = vi[vj]
    return zip_nt_params(model_params)

def timestamp(main, prefix = 'M'):
    if isinstance(main, str):
        return float(main[1:])
    return f'{prefix}{main:.2f}'