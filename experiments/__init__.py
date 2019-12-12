from itertools import product
from utils.param_ops import filter_flatten
from os import listdir

types = tuple(f[:-3] for f in listdir('experiments') if f.startswith('t_') and f.endswith('.py'))

data_settings = ['penn_settings', 'stan_settings']

def __para_to_str(pp):
    def single(p):
        if callable(p):# class is also callable type(p) is type:
            return p.__name__
        elif isinstance(p, bool):
            return 'T' if p else 'F'
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, str):
            return p
        elif p is None:
            return '-'
        else:
            raise TypeError(f'Cannot handel {p}')
        # end of level 2
    # if isinstance(pp, tuple):
    #     pp = 
    # else:
    #     pp = single(pp)
    return ','.join(single(p) for p in pp)

def get_variations(task_variation, var_name):
    '''A task can be conducted in many actual ways (experiments).
    This ways are predefined in the task module through the
    .task_variation. And this variant is the elements of all
    possible independent hyper params. Thus, this function calculates
    the full combination of all these params for one way
    (an experiment / a model variant) and name the actual
    configuration to provide a edible user interface and
    consistant automatic management.'''
    kw_func = []
    kw_para = []
    if callable(task_variation[var_name][0]):
        skip = 1
        is_valid = task_variation[var_name][0]
    else:
        skip = 0
        is_valid = lambda kwfunc_params: True
    for func, params in task_variation[var_name][skip:]: # func0: (params, params), func1: (...
        kw_func.append(func)       # [func0,      func1,      ...]
        kw_para.append(params)     # [(pa,pb,..), (pA,pB,..), ...]
    for px in product(*kw_para):   # [(pa, pA, ..), (pb, pB, ..)]
        kwfunc_params = []
        kwfunc_disply = []
        for f, pp in zip(kw_func, px): # [(func0, pa), (func1, pA), ...] ...
            kwfunc_disply.append(f'{f.__name__}' + f'({__para_to_str(pp)})')
            kwfunc_params.append((f, pp))
        if is_valid(kwfunc_params):
            yield '.'.join(kwfunc_disply), kwfunc_params # 'rnn_var' 'predict(B.1...).' [(func0, pa), (func1, pA), ...]