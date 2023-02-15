from math import exp
from torch import optim, Tensor
from utils.shell_io import byte_style
from utils import do_nothing
from os.path import join

make_tensors = lambda *args, fn = do_nothing: tuple(fn(x).cpu().numpy() if isinstance(x, Tensor) else x for x in args)

class WarmOptimHelper:
    @classmethod
    def adam(cls, model, *args, **kwargs):
        opt = optim.Adam(model.parameters(), betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6)
        return cls(opt, *args, **kwargs)

    def __init__(self, optimizer, base_lr = 0.001, wander_threshold = 0.15, damp_threshold = 0.4, damp = 1):
        assert base_lr > 0
        assert 0 < damp <= 1
        assert 0 < wander_threshold < damp_threshold <= 1
        self._opt_threshold_damp = optimizer, wander_threshold, damp_threshold, damp
        self._base_lr_last_wr = base_lr, 0
        
    def __call__(self, epoch, wander_ratio, lr_factor = 1):
        opt, wander_threshold, damp_threshold, damp = self._opt_threshold_damp
        base_lr, last_wr = self._base_lr_last_wr
        if wander_ratio < damp_threshold < last_wr:
            base_lr *= damp
        self._base_lr_last_wr = base_lr, wander_ratio

        if wander_ratio < wander_threshold:
            learning_rate = base_lr * (1 - exp(- epoch))
        else:
            linear_dec = (1 - (wander_ratio - wander_threshold) / (1 - wander_threshold + 1e-20))
            learning_rate = base_lr * linear_dec
        learning_rate *= lr_factor

        for opg in opt.param_groups:
            opg['lr'] = learning_rate
        # self._lr_discount_rate = 0.0001
        # for params in self._model.parameters():
        #     if len(params.shape) > 1:
        #         nn.init.xavier_uniform_(params)
        return learning_rate
            
    @property
    def optimizer(self):
        return self._opt_threshold_damp[0]

def speed_logg(count, seconds, dm):
    speed_ba = count / seconds
    speed_dm = None
    base = f' {speed_ba:.1f}'

    if dm is None:
        desc = logg = ''
        base += f' sps. ({seconds:.3f})'
    else:
        dm_seconds = dm.duration
        speed_dm = count / dm_seconds
        logg = f' ◇ {speed_dm:.1f} sps'
        desc = byte_style(logg, '2')
        base += logg + f' ({seconds:.3f} ◇ {dm_seconds:.3f})'

    return desc, logg, speed_ba, speed_dm

def continuous_score_desc_logg(scores):
    desc_for_screen = '(' + pmr_f(scores["LR"] - scores["LP"], scores["F1"]) + ')'
    desc_for_logger = f'({scores["LP"]}/{scores["LR"]}/{scores["F1"]})'
    return scores, desc_for_screen, desc_for_logger

def pmr_f(pmr, f):
    desc_for_screen = byte_style('㏚', '3' if pmr > 0 else '6')
    desc_for_screen += f'{pmr:+.2f}/' + byte_style(f'{f:.2f}', underlined = True)
    return desc_for_screen

def discontinuous_score_desc_logg(tp, tr, tf, dp, dr, df):
    desc_for_screen = '(' + pmr_f(tr - tp, tf) + '|' + pmr_f(dr - dp, df) + ')'
    desc_for_logger = f'({tp:.2f}/{tr:.2f}/{tf:.2f}|{dp:.2f}/{dr:.2f}/{df:.2f})'
    return desc_for_screen, desc_for_logger

def sentiment_score_desc_logg(root_scores, all_scores):
    logg = []
    desc = '('
    scores = {}
    for field, root_value, all_value in zip(root_scores._fields, root_scores, all_scores):
        if field == 'q':
            value = f'{root_value:.0f}\'' + byte_style(f'{all_value:.2f}', '7', underlined = True)
        else:
            value = f'{root_value:.0f}\'{all_value:.0f}/'
        cap_field = field.upper()
        desc += byte_style(cap_field, '7') + value
        scores[cap_field] = root_value
        scores[field] = all_value
        logg.append(cap_field + f'{all_value:.2f}\'{root_value:.2f}')
    desc += ')'
    logg = '/'.join(logg)
    return scores, desc, logg

def serialize_matrix(m, skip = None):
    for rid, row in enumerate(m):
        offset = 0 if skip is None else (rid + skip)
        for cid, val in enumerate(row[offset:]):
            yield rid, offset + cid, float(val)

def sort_matrix(m, lhs, rhs, higher_better):
    lines = []
    n = max(len(n) for n in lhs) + 1
    for rid, row in enumerate(m):
        line = []
        for cid, _ in sorted(enumerate(row), key = lambda x: x[1], reverse = higher_better):
            line.append(rhs[cid])
        lines.append(lhs[rid].ljust(n) + ': ' + ' '.join(line))
    return '\n'.join(lines)

def save_txt(fname, append, lhv, rhv, dst, cos):
    with open(fname, ('w', 'a+')[append]) as fw:
        if append:
            fw.write('\n\n')
            lhv, rhv = lhv.label, rhv.label
            n = 'Label'
        else:
            lhv, rhv = lhv.tag, rhv.tag
            n = 'Tag'
        fw.write(f'Distance\n  {n}:\n')
        fw.write(sort_matrix(dst, lhv, rhv, False))
        fw.write(f'\nCosine\n  {n}:\n')
        fw.write(sort_matrix(cos, lhv, rhv, True))

def write_multilingual(work_dir, i2vs, model):
    # save vocabulary
    for corp, _i2vs in i2vs.items():
        with open(join(work_dir, 'tag.' + corp), 'w') as fw:
            fw.write('\n'.join(_i2vs.tag))
        with open(join(work_dir, 'label.' + corp), 'w') as fw:
            fw.write('\n'.join(_i2vs.label))
    # save Tag/Label
    for prefix in ('tag', 'label'):
        if get_label := prefix == 'label':
            fn = model.get_multilingual_label_matrices
        else:
            fn = model.get_multilingual_tag_matrices
        for lhs, rhs, dst, cos in fn():
            # save matrix
            #  # 'a+' if get_label else 'w' lhv, rhv = self.i2vs[lhs], self.i2vs[rhs]
            if lhs == rhs:
                fname = lhs
                save_txt(join(work_dir, lhs + '.txt'), get_label, i2vs[lhs], i2vs[lhs], dst, cos)
            else:
                fname = lhs + '.' + rhs
                save_txt(join(work_dir, lhs + '.' + rhs + '.txt'), get_label, i2vs[lhs], i2vs[rhs], dst, cos)
                save_txt(join(work_dir, rhs + '.' + lhs + '.txt'), get_label, i2vs[rhs], i2vs[lhs], dst.T, cos.T)
            with open(join(work_dir, prefix + '.' + fname + '.csv'), 'w') as fw:
                fw.write('type,row,col,value\n')
                if lhs == rhs:
                    for r, c, v in serialize_matrix(dst, 1):
                        fw.write(f'd,{r},{c},{v}\n')
                    for r, c, v in serialize_matrix(cos, 1):
                        fw.write(f'c,{c},{r},{v}\n')
                else:
                    for r, c, v in serialize_matrix(dst):
                        fw.write(f'd,{r},{c},{v}\n')
                    for r, c, v in serialize_matrix(cos):
                        fw.write(f'c,{r},{c},{v}\n')