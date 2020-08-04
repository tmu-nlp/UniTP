
from datetime import datetime
from utils.file_io import join, create_join, listdir, isdir, isfile, remove, rm_rf, rename
from utils.file_io import DelayedKeyboardInterrupt, copy_with_prefix_and_rename
from utils.yaml_io import load_yaml, save_yaml
from utils.param_ops import zip_nt_params, dict_print, change_key
from sys import stderr
from itertools import count
import torch

_rt_file = 'register_and_tests.yaml'
_rt_lock = 'register_and_tests.lock'
_sv_file = 'settings_and_validation.yaml'
_sv_lock = 'settings_and_validation.lock'

class Recorder:
    '''A Recorder provides environment for an Operator, created in a Manager, operated by the Operator.'''
    
    def __init__(self, task_dir, task_module, config_dict_or_instance, instance_name = None, keep_top_k = 4, evalb = None):
        def create_sv_file_lock(instance_dir):
            sv_file = join(instance_dir, _sv_file)
            sv_lock = join(instance_dir, _sv_lock)
            return sv_file, sv_lock
        # with DelayedKeyboardInterrupt():
        rt_file = join(task_dir, _rt_file)
        rt_lock = join(task_dir, _rt_lock)
        if isinstance(config_dict_or_instance, dict):
            rt, unlock = load_yaml(rt_file, rt_lock, True)
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
            config_dict_or_instance['results'] = {}
            sv_file, sv_lock = create_sv_file_lock(instance_dir)
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
            sv_file, sv_lock = create_sv_file_lock(instance_dir)
            assert isfile(sv_file), f"'{sv_file}' is not found."

        print_args = dict(flush = True, file = open(join(instance_dir, 'experiment.log'), 'a+'))
        print(datetime.now(), **print_args)
        self._print_args = print_args
        self._instance_dir = instance, instance_dir
        self._module     = task_module
        self._ckpt_fname = join(instance_dir, 'checkpoint')
        self._model_dir  = create_join(instance_dir, 'models')
        _, self._sv_unlock = load_yaml(sv_file, sv_lock, True)
        self._rt_file_lock = rt_file, rt_lock
        self._sv_file_lock = sv_file, sv_lock
        self._key = None
        self._cleaners = []
        self._keep_top_k = keep_top_k
        self._evalb = evalb

    def __del__(self):
        if isdir(self._instance_dir[1]):
            self._print_args['file'].close() # critical zone
            self._sv_unlock()
        # if input('*** Remove Experiment ? [n/N or any key] *** ').lower() != 'n':

    def delete_all(self):
        self._print_args['file'].close() # critical zone
        instance, instance_dir = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt.pop(instance)
        save_yaml(rt, *self._rt_file_lock)
        rm_rf(instance_dir, stderr)

    def log(self, *args, **kwargs):
        print(*args, **self._print_args, **kwargs)

    @staticmethod
    def msg(*args, **kwargs):
        print(*args, **kwargs, file = stderr, flush = True)

    def task_specs(self):
        from utils.param_ops import HParams
        specs = load_yaml(*self._sv_file_lock, wait_lock = False)
        _, model_type, train_type = self._module.get_configs()
        model_config = get_obj_from_config(model_type, specs['model'])
        train_config = get_obj_from_config(train_type, specs['train'])
        train_config = HParams(train_config)
        return specs['data'], model_config, train_config, specs['results']

    def create_join(self, *args):
        _, instance_dir = self._instance_dir
        return create_join(instance_dir, *args)

    def initial_or_restore(self, model, optimizer = None, restore_from_best_validation = False):
        model_fname = None
        if not restore_from_best_validation and isfile(self._ckpt_fname):
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

        elif isdir(self._model_dir) or restore_from_best_validation:
            resutls = load_yaml(*self._sv_file_lock, wait_lock = False)['results']
            if resutls:
                best_model = max(resutls, key = lambda x: resutls[x])
                model_fname = join(self._model_dir, best_model)

        if model_fname is None:
            epoch = global_step = 0
            fine_validation = False
            md = dict(model.named_parameters())
            print(dict_print(zip_nt_params(md), v_to_str = lambda tensor: '*'.join(str(s) for s in tensor.shape)), **self._print_args)
            total = 0
            for t in md.values():
                x = 1
                for s in t.shape:
                    x *= s
                total += x
            print('Total:', total, **self._print_args)
        else:
            checkpoint = torch.load(model_fname)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model_old_dict = checkpoint['model_state_dict']
                model_new_dict = model.state_dict()
                new_keys = tuple(model_new_dict)
                from utils.shell_io import byte_style
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
                        if match_depth > 1:
                            new_candidates[new_key] = match_depth
                    new_candidates = sorted(new_candidates, key = new_candidates.get, reverse = True)
                    if len(new_candidates) == 1:
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
                    else:
                        prompt = f'Change {old_key} into:\n'
                        for i, k in enumerate(new_candidates):
                            prompt += f'{i}) {k}\n'
                        new_key = input(prompt)
                        if new_key == 'q':
                            exit()
                        new_key = int(new_key)
                        assert new_key in range(len(new_candidates))
                        new_key = new_candidates[new_key]
                    change_key(model_old_dict, old_key, new_key)
                model.load_state_dict(checkpoint['model_state_dict'])
                decision = input(f'Save change to {model_fname}? [Y]')
                if decision == 'Y':
                    torch.save(checkpoint, model_fname)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch, fine_validation, global_step = checkpoint['status']
            self._key = checkpoint['key']
            
            print(f"Model restored from", model_fname, **self._print_args)
            Recorder.msg(f'Model Restored at {epoch:.2f}, scoring {self._key:.2f}')
            if restore_from_best_validation:
                return epoch, global_step
            epoch = int(epoch)
            # for cleaner in self._cleaners: # vis
            #     cleaner(after = epoch)
        return epoch, fine_validation, global_step

    def check_betterment(self, epoch, falling, global_step, model, optimizer, key):
        specs = load_yaml(*self._sv_file_lock, wait_lock = False)
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
            specs['results'][model_fname] = key
            results = specs['results']
            if len(results) > self._keep_top_k:
                weakest_model = min(results, key = lambda x: results[x])
                remove(join(self._model_dir, weakest_model))
                results.pop(weakest_model)
                print(' Replace worst model', weakest_model, 'with a', 'new best' if betterment else 'better', 'model', model_fname, **self._print_args)
            else:
                print(' A new', 'best' if betterment else 'better', 'model', model_fname, **self._print_args)
            save_yaml(specs, *self._sv_file_lock, wait_lock = False)
        else:
            print(**self._print_args)
        return betterment

    def register_test_scores(self, scores):
        instance, _ = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt[instance] = scores
        save_yaml(rt, *self._rt_file_lock)

    @staticmethod
    def experiments_status(task_path):
        rt_file = join(task_path, _rt_file)
        rt_lock = join(task_path, _rt_lock)
        (instance_status, unlock), modifed = load_yaml(rt_file, rt_lock, True), False
        status = dict(locking = [], unlocked = [], other = [])
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
                rename(fpath, new_fpath)
                fpath = new_fpath + '\t<- ' + folder
                instance = _instance
                modifed = True
            key = instance_status[instance].get('key', '?????')
            status['unlocked'].append(f'({key})\t {fpath}')

        unlock()
        if modifed:
            save_yaml(instance_status, rt_file, rt_lock)
        return status

    @property
    def evalb(self):
        return self._evalb

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