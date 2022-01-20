from utils.yaml_io import load_yaml
from utils.file_io import isdir, isfile, join, listdir, create_join, dirname
from utils.param_ops import get_sole_key
from sys import argv

_sv_file = 'settings_and_validation.yaml'
_, trial_dir = argv
instance_dir = dirname(trial_dir)
is_dccp = '_dccp/' in instance_dir
assert is_dccp or '_xccp/' in instance_dir
rt_file = join(trial_dir, 'register_and_tests.yaml')
sv_file = join(instance_dir, _sv_file)
assert isdir(trial_dir) and isfile(rt_file) and isfile(sv_file)

rt = load_yaml(rt_file, None, wait = False)
sv = load_yaml(sv_file, None, wait = False)
data = sv['data']
corp_name = get_sole_key(data)
data = data[corp_name]
baseline = sv['results']
basestep = float(max(baseline, key = baseline.get)[1:])
scores = set({'TF', 'DF'})
optuna_hyper = 'optuna', ('db' if is_dccp else 'dm'), corp_name, 'csv'
optuna_dtdif = ('diff', ) + optuna_hyper
diff_seq = []
count = 0
tfmax = 0
tfmin = 100
tfstr = None

csv_dir = create_join('R_ggplot', 'stat.model')
with open(join(csv_dir, '.'.join(optuna_hyper)), 'w') as fwh,\
     open(join(csv_dir, '.'.join(optuna_dtdif)), 'w') as fwd:
    common_head = 'tag', 'label', 'joint'
    if is_dccp:
        # has_shuffle = data['shuffle_swap'] is not None
        losses = common_head + ('_right', 'shuffled')
        model_head = 'orient', 'shuffle'
        binarization = 'head', 'left', 'midin25', 'midin50', 'midin75', 'right'
        factor_head = tuple('b.' + x for x in binarization)
    else:
        # has_neg = sv['train']['disco_2d_negrate'] > 0
        # losses = common_head[:-1] + ('fence', 'disco_1d', 'disco_2d', 'disco_2d_neg')
        # model_head = ('disc', 'biaff', 'neg')
        losses = common_head[:-1] + ('fence', 'disco_1d', 'disco_2d', 'disco_2d_intra', 'disco_2d_inter')
        model_head = ('disc', 'biaff', 'disco_2d_intra', 'disco_2d_inter')
        medoids = 'head', 'continuous', 'left', 'random', 'right'
        factor_head = tuple('r.' + x for x in ('sub', 'intra', 'inter') + medoids)
    head_loss = tuple('l.' + x for x in common_head + model_head)
    fwh.write(','.join(head_loss + factor_head) + ',lr,tf,df\n')
    fwd.write('dev.tf,test.tf,test.df,n.step\n')
    head_len = len(head_loss + factor_head) + 1

    for fname in listdir(trial_dir):
        fpath = join(trial_dir, fname)
        if not isdir(fpath):
            continue
        inst_id, name = fname.split('.', 1)
        if scores - rt[inst_id].keys():
            print('Skip', fname)
            continue
        trial_sv = load_yaml(join(trial_dir, fname, _sv_file), None, False)
        trial_loss = trial_sv['train']['loss_weight']
        trial_data = trial_sv['data'][corp_name]
        values = [trial_loss[x] for x in losses]
        
        if is_dccp:
            trial_factors= trial_data['binarization']
            values.extend(trial_factors[x] for x in binarization)
        else:
            trial_factors= trial_data['medium_factor']
            values.append(trial_factors['balanced'])
            values.append(trial_sv['train']['disco_2d_intra_rate'])
            values.append(trial_sv['train']['disco_2d_inter_rate'])
            values.extend(trial_factors['others'][x] for x in (medoids))
        values.append(trial_sv['train']['learning_rate'])
        assert head_len == len(values)
        if (tf := rt[inst_id]['TF']) > tfmax:
            tfmax = tf
            tfstr = fname
        if tf < tfmin:
            tfmin = tf
        df = rt[inst_id]['DF']
        values.append(tf)
        values.append(df)
        fwh.write(','.join(str(x) for x in values) + '\n')
        final_step = trial_sv['results']
        final_step, dev_tf = max(final_step.items(), key = lambda x: x[1])
        final_step = float(final_step[1:])
        fwd.write(f'{dev_tf},{tf},{df},{final_step - basestep}\n')
        diff_seq.append(dev_tf - tf)
        count += 1

mean_diff = sum(diff_seq) / len(diff_seq)
print(f'{count} finished trials - w/ max TF {tfmax}(+{tfmax-tfmin:.2f}); {tfstr} - {mean_diff:.2f}')