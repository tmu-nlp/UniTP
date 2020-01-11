#!/usr/bin/env python

import os
import sys
import argparse
import importlib
import data
import experiments
from os import mkdir, listdir
from os.path import isdir, isfile, join
from utils.yaml_io import save_yaml, load_yaml
from utils.file_io import create_join, DelayedKeyboardInterrupt
from utils.types import valid_size, fill_placeholder
from utils.param_ops import zip_nt_params, unzip_nt_params, iter_zipped_nt_params, change_key, dict_print
from utils.str_ops import strange_to
from utils.shell_io import call_fasttext
from collections import defaultdict
from datetime import datetime
from data.penn_types import E_PENN, C_ABSTRACT
import pdb

_mfile = 'manager.yaml'
_lfile = 'manager.lock'
_rfile = 'register_and_results.yaml'
_rlock = 'register_and_results.lock'
_data_basic_info = {'source_path': fill_placeholder,
                    'local_path' : None,
                    'valid_sizes': None}

def _new_status():
    '''This function provides a dynamic basis of the framework.
    It is used by the class Manager to scan the status of the
    project.'''

    tool = dict(fasttext = dict(path = fill_placeholder + 'fasttext',
                                ft_bin = dict(en = fill_placeholder + 'wiki.en.bin',
                                              zh = fill_placeholder + 'cc.zh.300.bin',
                                              ja = fill_placeholder + 'cc.ja.300.bin'),
                                ft_lower = False),
                evalb = dict(path = fill_placeholder + 'evalb',
                             prm  = fill_placeholder + 'default.prm'))
    data_status = {}
    dat_modules = {}
    for module_name in data.types:
        m = importlib.import_module(f'data.{module_name}')
        for corp_name, info in m.build_params.items():
            d = _data_basic_info.copy()
            d['build_params'] = info
            data_status[corp_name] = d
            dat_modules[corp_name] = m

    task_status = {}
    exp_modules = {}
    for module_name in experiments.types:
        m = importlib.import_module(f'experiments.{module_name}')
        t = {}
        data_type, model_type = m.get_configs()
        t['data']  = zip_nt_params((k,v.default) for k,v in iter_zipped_nt_params(data_type))
        t['model'] = zip_nt_params((k,v.default) for k,v in iter_zipped_nt_params(model_type))
        module_name = module_name[2:]
        task_status[module_name] = t
        exp_modules[module_name] = m

    manager_status = {'tool': tool,
                      'data': data_status,
                      'task': task_status}
    return manager_status, exp_modules, dat_modules

def _recursive_fill(base, fresh, overwrite = False, serious_check = True):
    '''A handy function to add missing slots to base
    Place holder None will not overwrite any value, unless
    ``overwrite = True`` when you're sure your value is absolutely proper.
    ``serious_check == False`` is used to get rid of the errors from the annotations'''
    # valid_sizes: null <- (1,3,4) <- (3, 4)
    # vocab_size_: 3 <- null <- (3, 4) # can be rewrite to null for special purpose? or -1 is okay for placeholder

    for k, v in fresh.items(): # from fresh to base (update)
        if k not in base:
            base[k] = v
        elif overwrite or base[k] is None and v:
            base[k] = v # TODO: find a simpler rule for all parts of the tree
        elif isinstance(v, dict) and isinstance(base[k], dict):
            _recursive_fill(base[k], v, overwrite, serious_check)
        elif v is None:
            continue
        elif serious_check and base[k] is not None and base[k] != v:
            raise ValueError(f"Overwrite base[{k}] '{base[k]}' with '{v}'.")

def _recursive_take(furui, standard):
    '''Another handy funtion to perform 'furui - standard'
    This function only check tree nodes instead of actual values,
    which are the stern of the framework.'''
    obsolete = {}
    for k in furui.keys() - standard.keys(): # redundants
        obsolete[k] = furui.pop(k)

    for k, v in furui.items():
        if isinstance(v, dict) and isinstance(standard.get(k, None), dict):
            r = _recursive_take(v, standard[k])
            if r: obsolete[k] = r
    return obsolete

class Manager:
    '''The protocal of the project.
    Check the validity of data and experiments.

    When data is ready, and also experiments are ready,
    The experiments (sub missions of tasks) can be listed
    to run.

    A brief progress will be presented in a yaml file.
    You are also recommended to run experiments through
    this Manager, which garentees the scale of all things.'''
    def __init__(self, work_dir, overwrite = False):
        mfile = join(work_dir, _mfile)
        lfile = join(work_dir, _lfile)
        self._mfile_lfile = mfile, lfile
        self._work_dir = work_dir

        basic_status, self._exp_modules, self._dat_modules = _new_status()
        if isdir(work_dir):
            if isfile(mfile):
                status = load_yaml(*self._mfile_lfile)
                obsolete = _recursive_take(status, basic_status)
                if obsolete:
                    ofile = join(work_dir, 'obsolete.yaml')
                    print(f"Removing obsolete settings to: '{ofile}'", file = sys.stderr)
                    save_yaml(obsolete, ofile, None)
                _recursive_fill(status, basic_status, overwrite, serious_check = False)
            else:
                status = basic_status
                if len(listdir(work_dir)):
                    print(f"CREATING WARNING: work_dir '{work_dir}' is NOT empty.", file = sys.stderr)
            save_yaml(status, *self._mfile_lfile) # save for new appended feature (experiments)
        else:
            mkdir(work_dir)
            status = basic_status
            save_yaml(status, *self._mfile_lfile)

    def list_experiments_status(self, print_file = sys.stdout):
        from utils.recorder import Recorder
        task_status = load_yaml(*self._mfile_lfile)['task']
        for task_name in task_status:
            task_path = join(self._work_dir, task_name)
            if not isdir(task_path):
                continue
            print(f'In task ==={task_name}===', file = print_file)
            status = Recorder.experiments_status(task_path)
            for key, vlist in status.items():
                if vlist:
                    print(f'  {key}:', file = print_file)
                    for info in vlist:
                        print(info, file = print_file)

    def check_data(self, build_if_not_yet = False, num_thread = 1):
        modified = []
        ready_paths = {}
        verbose = defaultdict(list)
        status = load_yaml(*self._mfile_lfile)
        tools  = status['tool']
        data_status   = status['data']
        
        if isfile(tools['fasttext']['path']):
            fasttext = tools['fasttext']
        else:
            fasttext = None
            verbose['fasttext'].append('Invalid path for executive fasttext')

        for corp_name, datum in data_status.items():
            # import pdb; pdb.set_trace()
            sp = datum['source_path']
            if not isinstance(sp, str) or not isdir(sp):
                verbose[corp_name].append(f'Invalid source_path')
                continue

            lp = datum['local_path']
            lp = join('data', corp_name) if lp is None else lp.strip()
            elp = join(self._work_dir, lp)
            if not isdir(elp):
                lp = join('data', corp_name)
                elp = join(self._work_dir, lp)
                
            m = self._dat_modules[corp_name]
            ft_ready = m.check_fasttext(elp)
            ds_ready = False
            sizes = datum['valid_sizes']
            if sizes:
                try:
                    sizes = strange_to(sizes)
                except Exception as e:
                    sizes = None
                    print(e)
            if datum['valid_sizes'] and sizes and m.check_data(elp, sizes):
                ds_ready = True
                if ft_ready:
                    ready_paths[corp_name] = elp
                    print(f"*local dataset '{corp_name}' is ready", file = sys.stderr)
                    continue # ready, no need to build
            elif build_if_not_yet:
                print(f"(Re)build local dataset '{corp_name}'", file = sys.stderr)
                # try:
                sizes = m.build(create_join(self._work_dir, 'data', corp_name), sp, corp_name,
                                **datum['build_params'], num_thread = num_thread)
                ft_ready = False
                ds_ready = True
                # except Exception as e:
                #     verbose[corp_name].append(f'Build Error: {str(e)}')
                #     continue
                
                datum['local_path'] = lp
                datum['valid_sizes'] = ','.join(str(i) for i in sizes)
                modified.append(corp_name)
                ready_paths[corp_name] = elp
            else:
                verbose[corp_name].append(f'Ready to build')

            if fasttext and ds_ready and not ft_ready:
                m.call_fasttext(fasttext, elp, corp_name)

        if verbose:
            print('data:', file = sys.stderr)
            for data, msg in verbose.items():
                print(data.rjust(15), '; '.join(msg) + '.', file = sys.stderr)
        if modified:
            # reload for consistency, update only modified data
            file_status = load_yaml(*self._mfile_lfile)
            for m in modified:
                _recursive_fill(file_status['data'][m], data_status[m], True)
            save_yaml(file_status, *self._mfile_lfile)
        return ready_paths, status

    def check_task_settings(self):
        ready_tasks = {}
        ready_dpaths, status = self.check_data()
        verbose = {}
        evalb = status['tool']['evalb']

        evalb_errors = []
        if not isfile(evalb['path']):
            evalb_errors.append('Invalid evalb path')
        if not isfile(evalb['prm']):
            evalb_errors.append('Invalid evalb prm file')

        for module_name, task_config in status['task'].items():
            m = self._exp_modules[module_name]
            data_type, model_type = m.get_configs()
            data_config = task_config['data']
            errors = []

            if data_config.keys() > ready_dpaths.keys():
                errors.append('Data is not ready: ' + ', '.join(data_config.keys() - ready_dpaths.keys()))
            else:
                if any(d in E_PENN for d in data_config):
                    errors.extend(evalb_errors)
                for k, mnp, unp in iter_zipped_nt_params(data_type, data_config):
                    if not mnp.validate(unp):
                        errors.append(f'Invalid data_config: {k} = {unp}')
                    # try:
                    #     if not ( or isinstance(unp, int) and mnp.is_valid(mnp[unp])):
                            
                    # except Exception as e:
                    #     errors.append(f'{e} raised by {k}({mnp}) with input {unp}')

            for k, mnp, unp in iter_zipped_nt_params(model_type, task_config['model']):
                if not mnp.validate(unp): #( or isinstance(unp, int) and mnp.is_valid(mnp[unp])):
                    errors.append(f'Invalid model_config: {k} = {unp}')
            
            if errors:
                verbose[module_name] = errors
            else:
                ready_tasks[module_name] = task_config

        if verbose:
            print('task:', file = sys.stderr)
            for task_name, msg in verbose.items():
                print(task_name.rjust(15), '; '.join(msg) + '.', file = sys.stderr)

        return ready_dpaths, ready_tasks, status

    def ready_experiments(self, print_file = sys.stdout):
        ready_dpaths, ready_tasks, status = self.check_task_settings()
        print('Ready experiments:', ', '.join(ready_tasks.keys()), file = print_file)
        return ready_dpaths, ready_tasks, status

    def select_and_run(self, args):
        ready_paths, ready_tasks, status = self.check_task_settings()
        assert ready_tasks, 'No experiments ready :('

        from utils.mgr import check_select, check_resume_and_instances, check_train
        from utils.recorder import Recorder
        from utils.train_ops import train
        task, corp_name, name = check_select(args.select)
        resume, exp_ids = check_resume_and_instances(args.instance)
        
        module = self._exp_modules[task]
        task_spec = ready_tasks[task]
        data_config = task_spec['data']

        def diff_recorder(config_dict_or_instance):
            task_dir = create_join(self._work_dir, task)
            return Recorder(task_dir,
                            module,
                            config_dict_or_instance,
                            name,
                            evalb = status['tool']['evalb'])

        if resume or None in exp_ids:
            train_params = check_train(args.train)
        
        if None in exp_ids: # train new
            if corp_name in E_PENN: # only happen at penn data
                change_key(data_config, C_ABSTRACT, corp_name)
            for d, c in data_config.items():
                if d not in ready_paths:
                    print(f"{d} is an abstract data, you might mean: {' or '.join(ready_paths.keys())}", file = sys.stderr)
                    exit()
                if c is None:
                    data_config[d] = dict(data_path = ready_paths[d])
                else:
                    c['data_path'] = ready_paths[d]
                    if c.get('trapezoid_height', None) is not None: # a trigger for source corpus
                        corp_status = status['data'][d]
                        c['source_path'] = corp_status['source_path']
                        c['data_splits'] = corp_status['build_params']

        for exp_id in exp_ids:
            recorder = diff_recorder(task_spec) if exp_id is None else diff_recorder(exp_id)
            print(recorder.create_join())
            if exp_id is None or resume:
                try:
                    operator = module.get_configs(recorder)
                    recorder.register_test_scores(train(train_params, operator))
                except (Exception, KeyboardInterrupt) as e:
                    print(f'Cancel experiment ({recorder.create_join()})', file = sys.stderr)
                    if not isinstance(e, KeyboardInterrupt):
                        import traceback
                        traceback.print_exc()
                        
                    print(e, file = sys.stderr)
                    if input('Remove recorder ? [Any key or n]').lower() != 'n':
                        recorder.delete_all()
            else:
                recorder.register_test_scores(module.get_configs(recorder).test_model())

def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Manager', usage = '%(prog)s DIR [options]',
        description = 'A handy guider and manager for all the data and experiments',
        add_help    = True,
    )
    parser.add_argument('base', metavar = 'DIR', help = 'working directory', type = str)
    parser.add_argument('-R', '--reset',     help = 'initial manager.yaml', action = 'store_true', default = False)
    parser.add_argument('-p', '--prepare',   help = 'prepare all dataset for training', action = 'store_true', default = False)
    parser.add_argument('-P', '--threads',   help = 'a number of threads for pre-processing the data', type = int, default = -1)
    parser.add_argument('-m', '--menu',      help = 'list available sublayer configurations', action = 'store_true', default = False)
    parser.add_argument('-g', '--gpu',       help = 'pass to environment', type = str, default = '0')
    parser.add_argument('-x', '--train',     help = '[:max_epoch][>eval_skip][/eval_nth][|wander_stop][&test_with_eval]', type = str, default = '')
    parser.add_argument('-s', '--select',    help = 'select (a sub-layer config id)[/data][:folder] name to run', type = str)
    parser.add_argument('-i', '--instance',  help = 'test an trained model by the folder id without its suffix name', type = str)
    parser.add_argument('-b', '--beams',     help = 'beam sizes for test and infer', nargs = 2, type = int, default = [0, 0])
    parser.add_argument('-v', '--visualize', help = 'for parsing and its joint task, calc recall and f1, for others, make visualization and other detailded analysis') # , action = 'store_const', const = ''
    args = parser.parse_args()
    if args.base is None or not isdir(args.base):
        parser.print_help()
        print('[Please provide an working folder. You might need to use mkdir.]', file = sys.stderr)
        exit()
    return args

if __name__ == '__main__':
    args = get_args()
    manager = Manager(args.base, args.reset)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.threads > 0:
        from utils import types
        types.num_threads = args.threads
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if args.menu:# args.experiments
        manager.ready_experiments()
    elif args.select:
        manager.select_and_run(args)
        # except ValueError as ve:
        #     print('Dev: ValueError [Exit]')
        #     print(ve)
    else:
        manager.check_data(build_if_not_yet = args.prepare)
        with DelayedKeyboardInterrupt():
            manager.list_experiments_status() # refine this one