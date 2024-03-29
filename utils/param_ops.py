from itertools import product

def prod_flatten(*basis):
    basis = tuple(b if isinstance(b, tuple) else (b,) for b in basis)
    return tuple(flatten(param) for param in product(*basis))

def filter_flatten(flatten, is_valid):
    return tuple(item for item in flatten if is_valid(item))

def flatten(param):
    flat = []
    for x in param:
        if isinstance(x, tuple):
            flat.extend(flatten(x))
        else:
            flat.append(x)
    return tuple(flat)

class HParams:
    @staticmethod
    def check(dict_data, kwargs):
        if dict_data is None:
            dict_data = kwargs
        else:
            assert isinstance(dict_data, dict), '#1 should be a dict'
        if any('.' in k for k in dict_data.keys()):
            dict_data = zip_nt_params(dict_data)
        return dict_data

    def __init__(self, dict_data = None, fallback_to_none = False, **kwargs):
        self._fb2non = fallback_to_none
        self._nested = HParams.check(dict_data, kwargs)

    def create(self, dict_data = None, **kwargs):
        self._nested.update(HParams.check(dict_data, kwargs))

    def create_sub(self, name, dict_data = None, **kwargs):
        self._nested[name] = HParams.check(dict_data, kwargs)

    def __getattr__(self, attr_name):
        if self._fb2non and attr_name not in self._nested:
            return None
        val = self._nested[attr_name]
        if isinstance(val, dict):
            return HParams(val)
        return val

    def get(self, attr_name, fallback = None):
        if attr_name not in self._nested:
            return fallback
        return self.__getattr__(attr_name)

    def __str__(self):
        return dict_print(self._nested, 2)

from copy import deepcopy
def change_key(dict_data, from_key, *to_keys):
    value = dict_data.pop(from_key)
    for eid, to_key in enumerate(to_keys):
        if eid:
            dict_data[to_key] = deepcopy(value)
        else:
            dict_data[to_key] = value
    return value

def shift_key(key, from_dict, *to_dicts):
    value = from_dict.pop(key)
    for to_dict in to_dicts:
        to_dict[key] = value
    return value

def zip_nt_params(nt_params):
    _nt_params = {}
    if isinstance(nt_params, dict):
        nt_params = nt_params.items()
    for k, v in nt_params:
        if '.' in k:
            sub = _nt_params
            keys = k.split('.')
            klen = len(keys)
            # pdb.set_trace()
            for i, ki in enumerate(keys):
                if i == klen - 1: # terminal key
                    sub[ki] = v
                    break
                elif ki not in sub:
                    sub[ki] = {}
                sub = sub[ki]
        else:
            _nt_params[k] = v
    return _nt_params

def unzip_nt_params(nt_params):
    return {k:v for k,v in iter_zipped_nt_params(nt_params)}

def iter_zipped_nt_params(nt_params, ya_nt_params = None, ignore_missing_keys = False):
    for k, v in nt_params.items():
        if isinstance(v, dict):
            if ya_nt_params is None:
                for ki, vi in iter_zipped_nt_params(v):
                    yield k + '.' + ki, vi
            elif ignore_missing_keys and k not in ya_nt_params:
                continue
            else:
                for ki, vi, vj in iter_zipped_nt_params(v, ya_nt_params[k], ignore_missing_keys):
                    yield k + '.' + ki, vi, vj
        else:
            if ya_nt_params is None:
                yield k, v
            elif ignore_missing_keys and k not in ya_nt_params:
                continue
            else:
                yield k, v, ya_nt_params[k]


def dict_print(d, indent = 0, indent_inc = 2, indent_char = ' ', v_to_str = lambda v: f' {v}'):
    s = ''
    for k, v in d.items():
        s += indent_char * indent
        s += k + ':'
        if isinstance(v, dict):
            s += '\n' + dict_print(v, indent + indent_inc, indent_inc, indent_char, v_to_str)
        else:
            if isinstance(v, (list, tuple)):
                for i in v:
                    s += indent_char * indent + f' {i}\n'
            else:
                s += v_to_str(v)
        s += '\n'
    return s[:-1]

def fold_same_children(d, sep = '+'):
    groups = []
    for xk, xv in d.items():
        if any(xk in g for g in groups): continue
        g = set({xk})
        for yk, yv in d.items():
            if any(yk in g for g in groups): continue
            if xv == yv:
                g.add(yk)
        groups.append(g)
    c = {}
    for g in groups:
        sg = g
        if all(isinstance(k, int) or isinstance(k, str) and k.isdigit() for k in g):
            k = sorted(int(k) for k in g)
        k = sep.join(k if isinstance(k, str) else str(k) for k in sg)
        v = d[g.pop()]
        if not g and isinstance(v, dict): 
            v = fold_same_children(v)
        c[k] = v
    return c

def more_kwargs(base, **kwargs):
    for k,v in base.items():
        if k in kwargs:
            raise KeyError('Key in more is already exist: %r:%r(%r)', (k, v, base[k]))
        kwargs[k] = v
    return kwargs

def less_kwargs(base, key, default_value):
    if key in base:
        return base.pop(key)
    return default_value

def get_sole_key(dic):
    # prev iter & next was used, catch an exception is necessary.
    shake = set(dic)
    sole = shake.pop()
    assert not shake, f'{sole} is not the sole key in {dic}'
    return sole

def replace_args_kwargs(fn, n, _args, key, kw_args):
    m = n + 1
    if len(_args) >= m and _args[n] is not None:
        _args = _args[:n] + (fn(_args[n]),) + _args[m:]
    elif (val := kw_args.get(key)) is not None:
        kw_args[key] = fn(val)
    return _args