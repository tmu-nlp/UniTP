from utils.yaml_io import load_yaml
from utils.file_io import isdir, isfile, join, listdir, create_join, dirname, basename
from utils.param_ops import get_sole_key
from utils.recorder import summary_trials
from utils.types import F_RAND_CON, F_RAND_CON_SUB, F_RAND_CON_MSB, E_ORIF5_HEAD
from sys import argv

_rt_file = 'register_and_tests.yaml'
_sv_file = 'settings_and_validation.yaml'
_, trial_dir = argv
instance_dir = dirname(trial_dir)
is_dccp = '_dccp/' in instance_dir
inst_id = basename(instance_dir).split('.')[0]
base_rt = join(dirname(instance_dir), _rt_file)
rt_file = join(trial_dir, _rt_file)
sv_file = join(instance_dir, _sv_file)
assert isfile(base_rt)
assert isdir(trial_dir) and isfile(rt_file) and isfile(sv_file)

summary_trials(trial_dir, 'rank.txt')
if not (is_dccp or '_xccp/' in instance_dir):
    exit()

base_rt = load_yaml(base_rt, None, wait = False)[inst_id]
rt = load_yaml(rt_file, None, wait = False)
sv = load_yaml(sv_file, None, wait = False)
data = sv['data']
corp_name = get_sole_key(data)
data = data[corp_name]


model_head = 'tag', 'label'
if is_dccp:
    # has_shuffle = data['ply_shuffle'] is not None
    if is_dccp_con := data['binarization'][F_RAND_CON]:
        data_head = 'left, right', 'sub', 'msb'
    else:
        data_head = E_ORIF5_HEAD
    model_head = model_head + ('orient', 'joint', 'shuffled_joint', 'shuffled_orient')
else:
    medoids = 'head', 'continuous', 'left', 'random', 'right'
    data_head = 'sub', 'msb', 'intra_rate', 'inter_rate', 'max_inter_height'
    data_head = data_head + medoids
    model_head = model_head + ('fence', 'disco_1d', 'disco_2d', 'disco_2d_intra', 'disco_2d_inter')


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
base_tf = base_rt["TF"]

head = data_head + model_head
head_len = len(head)

csv_dir = create_join('R_ggplot', 'stat.model')
with open(join(csv_dir, '.'.join(optuna_hyper)), 'w') as fwh,\
     open(join(csv_dir, '.'.join(optuna_dtdif)), 'w') as fwd:
    fwh.write(','.join(head) + ',lr,dev.tf,tf,df,gain\n')
    fwd.write('dev.tf,test.tf,test.df,gain,n.step\n')

    for fname in listdir(trial_dir):
        fpath = join(trial_dir, fname)
        if not isdir(fpath):
            continue
        inst_id, name = fname.split('.', 1)
        if scores - rt[inst_id].keys():
            print('Skip', fname)
            continue
        trial_sv = load_yaml(join(trial_dir, fname, _sv_file), None, False)
        trial_data = trial_sv['data'][corp_name]
        trial_loss = trial_sv['train']['loss_weight']
        head_values = [trial_loss[x] for x in model_head]
        
        if is_dccp:
            trial_factors = trial_data['binarization']
            if is_dccp_con:
                data_values = [trial_factors[x] for x in (F_RAND_CON, F_RAND_CON_SUB, F_RAND_CON_MSB)]
            else:
                data_values = [trial_factors[x] for x in E_ORIF5_HEAD]
        else:
            trial_factors = trial_data['medium_factor']
            data_values = [trial_factors['balanced']]
            data_values.append(trial_factors['more_sub'])
            data_values.append(trial_sv['train']['disco_2d_intra_rate'])
            data_values.append(trial_sv['train']['disco_2d_inter_rate'])
            data_values.append(trial_data['max_inter_height'])
            data_values.extend(trial_factors['others'][x] for x in (medoids))
        values = data_values + head_values
        assert head_len == len(values)
        values.append(trial_sv['train']['learning_rate'])
        if (tf := rt[inst_id]['TF']) < tfmin:
            tfmin = tf
        if tfmax < tf:
            tfmax = tf
            tfstr = fname
        df = rt[inst_id]['DF']
        gain = tf - base_tf
        final_step = trial_sv['results']
        final_step, dev_tf = max(final_step.items(), key = lambda x: x[1])
        final_step = float(final_step[1:])
        values.append(dev_tf)
        values.append(tf)
        values.append(df)
        values.append(gain)
        fwh.write(','.join(str(x) for x in values) + '\n')
        fwd.write(f'{dev_tf},{tf},{df},{gain},{final_step - basestep}\n')
        diff_seq.append(dev_tf - tf)
        count += 1

print(f'{count} finished trials - Best TF {tfmax}(+{tfmax-base_tf:.2f}) - {tfstr}')
print(f'  Mean dev over test: {sum(diff_seq) / len(diff_seq):.2f}')