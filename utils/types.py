from copy import copy, deepcopy

class BaseType:
    def __init__(self, default_val, validator = None, default_set = None, as_exception = False, as_index = False):
        self._val_as_index = default_val, as_index, as_exception
        self._set = None if default_set is None else tuple(default_set)
        self._fallback = None
        if validator is None:
            if default_set is None: # uncountable value, float
                assert not as_index
                if as_exception:
                    cls = type(default_val)
                    self._valid = lambda x: isinstance(x, cls)
                else: # constant
                    self._valid = lambda x: x == default_val
            else:
                assert as_index
                assert isinstance(default_val, int) or default_val is None
                if as_exception: # (nn.LSTM, nn.GRU)
                    names_set = tuple(x.__name__ for x in default_set)
                    if default_val is None:
                        names_set = (None,) + names_set
                    self._valid = lambda x: x in names_set
                else: # ('CV', 'NV')
                    self._valid = lambda x: x in default_set
        else:
            if default_set is None: # uncountable value | default_val can be an exception
                assert not as_index
                if as_exception: # [0, 1) or None
                    self._valid = lambda x: validator(x) or x == default_val
                else: # [0, 1] at 0.2
                    assert validator(default_val)
                    self._valid = validator
            elif as_index:
                assert not as_exception
                self._valid = lambda x: (x in default_set or validator(x))
            else:
                if as_exception: # []
                    self._valid = lambda x: (x in default_set or validator(x) or x == default_val)
                else:
                    self._valid = lambda x: (x in default_set or validator(x))
                    assert self._valid(default_val)

    @property
    def default(self):
        default_val, as_index, as_exception = self._val_as_index
        if as_index and default_val is not None:
            return (self._set[default_val].__name__ if as_exception else self._set[default_val])
        return default_val

    def validate(self, val):
        # valid = self._valid(val)
        # if not valid and self._fallback is not None:
        #     return self._fallback.validate(val)
        # return valid
        return self._valid(val)

    # def set_fallback(self, btype):
    #     assert isinstance(btype, BaseType)
    #     self._fallback = btype

    def __getitem__(self, idx):
        default_val, as_index, as_exception = self._val_as_index
        if as_index and as_exception:
            for x in self._set:
                if x.__name__ == idx:
                    return x
            return default_val
        # elif self._fallback is not None:
        #     idx = self._fallback[idx]
        return idx

E_FT = (False, True)
E_LY = (1, 2, 4, 8)
E_MS = (2, 32, 64, 128)

frac_close     = lambda x: 0 <= x <= 1
frac_open_1    = lambda x: 0 <= x < 1
frac_open_0    = lambda x: 0 < x <= 1
valid_size      = lambda x: isinstance(x, int) and x > 0
valid_odd_size  = lambda x: valid_size(x) and x % 2 == 1
valid_even_size = lambda x: valid_size(x) and x % 2 == 0
valid_epoch     = lambda x: isinstance(x, int) and x >= 0
beam_size_exp  = BaseType(0, as_index = True, default_set = range(100))
false_type     = BaseType(False, as_exception = True)
true_type      = BaseType(True,  as_exception = True)
frac_1         = BaseType(0.1, validator = frac_open_1)
frac_2         = BaseType(0.2, validator = frac_open_1)
frac_3         = BaseType(0.3, validator = frac_open_1)
frac_4         = BaseType(0.4, validator = frac_open_1)
frac_5         = BaseType(0.5, validator = frac_open_1)
frac_7         = BaseType(0.7, validator = frac_open_1)
frac_06        = BaseType(0.06, validator = frac_open_1)
rate_5         = BaseType(0.5, validator = frac_close)
distance_type  = BaseType(3.1, validator = lambda d: d > 0)
non0_5         = BaseType(0.5, validator = frac_open_0)
none_type      = BaseType(None)
num_ctx_layer = BaseType(8, validator = lambda x: isinstance(x, int) and 0 <= x <= 24)
num_ori_layer = BaseType(1, validator = lambda x: isinstance(x, int) and 1 <= x <= 4)
vocab_size = BaseType(None, validator = lambda x: isinstance(x, int) and 2 < x, as_exception = True)
word_dim   = BaseType(300, validator = valid_even_size)
orient_dim = BaseType(64,  validator = valid_even_size)
hidden_dim = BaseType(200, validator = valid_size)
train_batch_size = BaseType(80, validator = valid_size)
train_bucket_len = BaseType(4, validator = valid_epoch)
tune_epoch_type  = BaseType(None, as_exception = True, validator = valid_epoch)
train_max_len    = BaseType(None, validator = valid_size, as_exception = True)
fill_placeholder = '//FILL//THIS//'
trapezoid_height = BaseType(None, valid_size, as_exception = True)

NIL, UNK, BOS, EOS = '<nil>', '<unk>', '<bos>', '<eos>'
M_TRAIN = 'train'
M_DEVEL = 'devel'
M_TEST  = 'test'
M_INFER = 'infer'
E_MODE = (M_TRAIN, M_DEVEL, M_TEST, M_INFER)
E_ORIF = 'left', 'right', 'midin', 'midout'
O_LFT, O_RGT, O_MIN, O_MOT = E_ORIF
E_CNF = O_LFT, O_RGT

import os
num_threads = (os.cpu_count() - 2) if os.cpu_count() > 2 else 1

frac_7 = BaseType(0.7, frac_close)
frac_1 = BaseType(0.1, frac_close)
frac_3 = BaseType(0.3, frac_close)
binarization = {O_LFT: frac_7,
                O_RGT: frac_1,
                O_MIN: frac_1,
                O_MOT: frac_1}
binarization_cnf = {O_LFT: frac_7, O_RGT: frac_3}