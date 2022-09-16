#!/usr/bin/env python3
import sys
import argparse
import importlib
import data
import experiments
from os import mkdir, listdir, environ
from os.path import isdir, isfile, join, abspath
from utils.yaml_io import save_yaml, load_yaml
from utils.file_io import create_join, DelayedKeyboardInterrupt, link
from utils.types import fill_placeholder, K_CORP
from utils.param_ops import zip_nt_params, iter_zipped_nt_params, change_key
from utils.str_ops import strange_to
from utils.shell_io import byte_style
from collections import defaultdict

_mfile = 'manager.yaml'
_lfile = 'manager.lock'
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
                                              ja = fill_placeholder + 'cc.ja.300.bin',
                                              de = fill_placeholder + 'cc.de.300.bin',
                                              id = fill_placeholder + 'cc.id.300.bin'),
                                ft_lower = False),
                evalb = dict(path = fill_placeholder + 'evalb',
                             prm  = fill_placeholder + 'default.prm'),
                evalb_lcfrs_prm = 'discodop.prm')
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
        data_type, model_type, train_type = m.get_configs()
        t['data']  = zip_nt_params((k,v.default) for k,v in iter_zipped_nt_params(data_type))
        t['model'] = zip_nt_params((k,v.default) for k,v in iter_zipped_nt_params(model_type))
        t['train'] = zip_nt_params((k,v.default) for k,v in iter_zipped_nt_params(train_type))
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
        else:
            mkdir(work_dir)
            status = basic_status
        save_yaml(status, *self._mfile_lfile) # save for new appended feature (experiments)

    def list_experiments_status(self, print_file = sys.stdout):
        from utils.recorder import Recorder
        task_status = load_yaml(*self._mfile_lfile)['task']
        print(byte_style('::Experiment MONITOR::', '2'), file = print_file)
        for task_name in task_status:
            task_path = join(self._work_dir, task_name)
            if not isdir(task_path):
                continue
            print(f'In task <<< {byte_style(task_name, "6")} >>>', file = print_file)
            status = Recorder.experiments_status(task_path)
            for status_key, vlist in status.items():
                if vlist:
                    print(f'  {status_key}:', file = print_file)
                    for info in vlist:
                        print('    ' + info, file = print_file)

    def check_data(self, build_if_not_yet = False, num_thread = 1, print_file = sys.stderr):
        from data.io import check_fasttext
        modified = []
        ready_paths = {}
        verbose = defaultdict(list)
        status = load_yaml(*self._mfile_lfile)
        tools  = status['tool']
        data_status   = status['data']
        
        print(byte_style('::Data STATUS::', '2'), file = print_file)
        if isfile(tools['fasttext']['path']):
            fasttext = tools['fasttext']
        else: # TODO: make this an option.
            fasttext = None
            verbose['fasttext'].append('Invalid path for executive fasttext')

        for corp_name, datum in data_status.items():
            # import pdb; pdb.set_trace()
            sp = datum['source_path']
            if not isinstance(sp, str) or (not isdir(sp) if corp_name != 'tiger' else not isfile(sp)):
                verbose[corp_name].append(f'Invalid source_path {sp}')
                continue

            if isinstance(lp := datum['local_path'], str) and ((lp_exists := isdir(lp)) or isdir(join(self._work_dir, lp))):
                if lp_exists:
                    ep = lp
                else:
                    ep = join(self._work_dir, lp)
            else:
                lp = join('data', corp_name)
                ep = join(self._work_dir, lp)
                
            ds_ready = False
            ft_ready = check_fasttext(ep)
            m = self._dat_modules[corp_name]
            if sizes := datum['valid_sizes']:
                try:
                    sizes = strange_to(sizes)
                except Exception as e:
                    sizes = None
                    print(e)
            if datum['valid_sizes'] and sizes and m.check_data(ep, sizes):
                ds_ready = True
                if ft_ready:
                    ready_paths[corp_name] = ep
                    print(f"*corpus '{byte_style(corp_name, '3')}' is ready", file = print_file)
                    continue # ready, no need to build
            elif build_if_not_yet:
                print(f"(Re)build vocabulary for '{corp_name}'", file = print_file)
                # try:
                sizes = m.build(create_join(self._work_dir, 'data', corp_name), sp, corp_name,
                                **datum['build_params'], num_thread = num_thread)
                ft_ready = False
                ds_ready = True
                
                datum['local_path'] = lp
                datum['valid_sizes'] = ','.join(str(i) for i in sizes)
                modified.append(corp_name)
                ready_paths[corp_name] = ep
            else:
                verbose[corp_name].append(f'Ready to build')
                
            if fasttext and ds_ready and not ft_ready:
                m.call_fasttext(fasttext, ep, corp_name)

        if print_file and verbose:
            print('data:', file = print_file)
            for data, msg in verbose.items():
                print(data.rjust(15), byte_style('; '.join(msg), '1') + '.', file = print_file)
        if modified:
            # reload for consistency, update only modified data
            file_status = load_yaml(*self._mfile_lfile)
            for m in modified:
                _recursive_fill(file_status['data'][m], data_status[m], True)
            save_yaml(file_status, *self._mfile_lfile)
        return ready_paths, status

    def check_task_settings(self, print_file = sys.stderr):
        from data.io import check_fasttext
        ready_tasks = {}
        ready_dpaths, status = self.check_data()
        verbose = {}
        evalb = status['tool']['evalb']
        evalb_lcfrs = status['tool']['evalb_lcfrs_prm']

        evalb_errors = [] 
        if not isfile(evalb['path']):
            evalb_errors.append('Invalid evalb path')
        if not isfile(evalb['prm']):
            evalb_errors.append('Invalid evalb prm file')
        evalb_lcfrs_errors = []
        if not isfile(evalb_lcfrs):
            evalb_lcfrs_errors.append('Invalid evalb_lcfrs_prm file')

        for module_name, task_config in status['task'].items():
            m = self._exp_modules[module_name]
            data_type, model_type, train_type = m.get_configs()
            data_config = task_config['data']
            errors = []

            if 'sentiment' in module_name:
                if 'sstb' not in ready_dpaths:
                    errors.append('Core data \'sstb\' is not ready')
            elif 'tokenization' in module_name:
                if not ready_dpaths:
                    errors.append('None of the datasets is ready')
            else:
                if not (ready_dpaths.keys() & m.CORPORA):
                    errors.extend(evalb_errors)
                    errors.append('None of \'' + "', '".join(m.CORPORA) + '\' is ready')
                if all(not check_fasttext(path) for path in ready_dpaths.values()):
                    errors.append('Lack pre-trained embeddings')
                
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

            for k, mnp, unp in iter_zipped_nt_params(train_type, task_config['train']):
                if not mnp.validate(unp):
                    errors.append(f'Invalid train_config: {k} = {unp}')
                    
            if errors:
                verbose[module_name] = errors
            else:
                ready_tasks[module_name] = task_config

        if print_file and verbose:
            print('task:', file = print_file)
            for task_name, msg in sorted(verbose.items(), key = lambda x: x[0][::-1]):
                print(task_name.rjust(15), byte_style('; '.join(msg) + '.', '1'), file = print_file)

        return ready_dpaths, ready_tasks, status

    def ready_experiments(self, print_file = sys.stdout):
        ready_dpaths, ready_tasks, status = self.check_task_settings(None)
        print('Ready experiments:', ', '.join(byte_style(x, '3') for x in ready_tasks.keys()), file = print_file)
        return ready_dpaths, ready_tasks, status

    def select_and_run(self, args):
        ready_paths, ready_tasks, status = self.check_task_settings()
        assert ready_tasks, 'No experiments ready :('

        from utils.mgr import check_select, check_instances_operation, check_train
        from utils.recorder import Recorder
        from utils.train_ops import train
        task, corp_name, spec_name = check_select(args.select)
        op_code, exp_ids = check_instances_operation(args.instance)
        assert op_code in (None, False, 'r', 'd', 'f'), f'Unknown operation {op_code}, options are [r]esume [d]elete [f]ork'

        assert task in self._exp_modules, f'No such task module {task} in [' + ', '.join(self._exp_modules.keys()) + ']'
        assert task in ready_tasks, f'No such ready task_spec {task} in [' + ', '.join(ready_tasks.keys()) + ']'
        module = self._exp_modules[task]
        task_path = create_join(self._work_dir, task)
        task_spec = ready_tasks[task]
        data_config = task_spec['data']

        def diff_recorder(config_dict_or_instance):
            if task.endswith('cb') or task.endswith('cm') or task.endswith('_sentiment') or task.endswith('_ner'):
                evalb = status['tool']['evalb']
                evalb = abspath(evalb['path']), '-p', abspath(evalb['prm'])
            elif task.endswith('db') or task.endswith('dm'):
                evalb = abspath(status['tool']['evalb_lcfrs_prm'])
            else:
                evalb = None
            return Recorder(task_path,
                            module.get_configs,
                            config_dict_or_instance,
                            spec_name,
                            evalb = evalb)

        train_or_resume_training = op_code == 'r' or None in exp_ids
        if train_or_resume_training:
            train_params = check_train(args.train)
        
        if None in exp_ids: # train new
            if corp_name:
                expected_corps = set()
                for corp in corp_name.split(','):
                    assert corp in module.CORPORA
                    expected_corps.add(corp)
            else:
                expected_corps = module.CORPORA & ready_paths.keys()
            if (excluded_corps := module.CORPORA - expected_corps) and not corp_name:
                print(', '.join(byte_style(x, '1') for x in excluded_corps), byte_style('are not ready and thus excluded.', '3'))

            for corp in module.CORPORA:
                if corp in excluded_corps:
                    data_config[K_CORP].pop(corp)
                else:
                    data_config[K_CORP][corp].update(status['data'][corp])
                    data_config[K_CORP][corp].update(local_path = join(self._work_dir, status['data'][corp]['local_path']))

        for exp_id in exp_ids:
            recorder = diff_recorder(task_spec) if exp_id is None else diff_recorder(exp_id)
            print(recorder.create_join())

            if op_code == 'f':# and input(f'Fork to {recorder} ? [Y or any key]').lower() != 'Y':
                trail_folder = spec_name or 'trials'
                trial = recorder.best_trial(trail_folder)
                pid, path = recorder._instance_dir
                spec_name = f'{{{pid}.{trial.tid}}}'
                fr = Recorder(task_path,
                              module.get_configs,
                              trial.specs,
                              spec_name,
                              evalb = recorder.evalb)
                fr.detach()
                path = join(path, trail_folder, f'{trial.tid}.{trial.spec_string}', 'models')
                for model in listdir(path):
                    link(join(path, model), join(fr._model_dir, model))
                recorder.detach()
                continue

            if op_code == 'd' and input('Remove recorder ? [Y or any key]').lower() != 'Y':
                recorder.detach()
                recorder.delete_all()
                continue

            if train_or_resume_training:
                operator = None
                if args.tensorboard:
                    recorder.init_tensorboard()
                else:
                    recorder.msg('(Not tracking training statistics.)')
                try:
                    operator = module.get_configs(recorder)
                    recorder.register_test_scores(train(train_params, operator))
                except (Exception, KeyboardInterrupt) as e:
                    print(f'Cancel experiment ({recorder.create_join()})', file = sys.stderr)
                    if not isinstance(e, KeyboardInterrupt):
                        import traceback
                        traceback.print_exc()
                    print(e, file = sys.stderr)

                    if input('Remove recorder ? [Y or any key]').lower() == 'y':
                        recorder.delete_all()
                if operator:
                    operator.close()
            else:
                recorder.register_test_scores(module.get_configs(recorder).test_model())
            recorder.detach()

def get_args():
    parser = argparse.ArgumentParser(
        prog = 'Manager', usage = '%(prog)s DIR [options]',
        description = 'A handy guider and manager for all the data and experiments',
        add_help    = True,
    )
    parser.add_argument('base', metavar = 'DIR', help = 'working directory', type = str)
    parser.add_argument('-R', '--reset',       help = 'initial manager.yaml', action = 'store_true', default = False)
    parser.add_argument('-p', '--prepare',     help = 'prepare all dataset for training', action = 'store_true', default = False)
    parser.add_argument('-t', '--tensorboard', help = 'use tensorboard to track training stat', action = 'store_true', default = False)
    parser.add_argument('-T', '--threads',     help = 'a number of threads for pre-processing the data', type = int, default = -1)
    parser.add_argument('-g', '--gpu',         help = 'pass to environment', type = str, default = '0')
    parser.add_argument('-x', '--train',       help = 'fv=3:30:4,max=100,!,optuna [fine validation starts from the 3rd consecutive key score wandering, ends at the 30th wandering, occuring 4 times during one epoch. !test with devel set!]', type = str, default = 'mp')
    parser.add_argument('-s', '--select',      help = 'select (a sub-layer config id)[/data][:folder] name to run', type = str)
    parser.add_argument('-i', '--instance',    help = 'test an trained model by the folder id without its suffix name', type = str)
    parser.add_argument('-m', '--memory',      help = 'To preoccupy GPU memory in GB', type = int, default = None)
    args = parser.parse_args()
    if args.base is None or not isdir(args.base):
        parser.print_help()
        print('[Please provide an working folder. You might need to use mkdir.]', file = sys.stderr)
        exit()
    return args

if __name__ == '__main__':
    args = get_args()
    environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    manager = Manager(args.base, args.reset)

    import torch
    from utils import types
    types.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.threads > 0:
        types.num_threads = args.threads
    if args.memory:
        gb = int(args.memory * 1000 * 1000 * 1000)
        torch.empty(gb, dtype = torch.int8, device = types.device)
        print(f'Pre-occupy {gb:,} byte memory on GPU{args.gpu}')
    
    if args.prepare:
        manager.check_data(build_if_not_yet = True, print_file = sys.stderr if args.select else None)
    if args.select:
        manager.select_and_run(args)
        # except ValueError as ve:
        #     print('Dev: ValueError [Exit]')
        #     print(ve)
    else:
        with DelayedKeyboardInterrupt():
            manager.list_experiments_status() # refine this one
        manager.ready_experiments()