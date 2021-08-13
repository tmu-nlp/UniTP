import multiprocessing


def check_select(select):
    if ':' in select: # 3/ptb:annotate
        i = select.index(':')
        exp_name = select[i+1:]
        select = select[:i]
    else:
        exp_name = None

    if '/' in select:
        i = select.index('/')
        select, corp_name = select.split('/')
    else:
        corp_name = None
    return select, corp_name, exp_name

def check_instances_operation(instance):
    from utils.str_ops import strange_to
    op_code = instance and instance[0].isalpha() and instance[0]
    exp_ids = instance[1:] if op_code else instance
    exp_ids = strange_to(exp_ids, str) if exp_ids else [exp_ids]
    return op_code, exp_ids

def check_train(train_str):
    # fv=4:30:4,max=100,!
    train = dict(test_with_validation = False,
                 fine_validation_at_nth_wander = 10,
                 stop_at_nth_wander = 100,
                 fine_validation_each_nth_epoch = 4,
                 update_every_n_batch = 1,
                 optuna_trials = 0,
                 max_epoch = 1000,
                 multiprocessing_decode = False)
    assert ' ' not in train_str
    for group in train_str.split(','):
        if group.startswith('fine='):
            group = [int(x) if x else 0 for x in group[5:].split(':')]
            assert 1 <= len(group) <= 3
            if group[0]:
                train['fine_validation_at_nth_wander'] = n = group[0]
                assert n >= 0, 'fine_validation_at_nth_wander < 0'
            if len(group) > 1 and group[1]:
                train['stop_at_nth_wander'] = n = group[1]
                assert n >= 0, 'stop_at_nth_wander < 0'
            if len(group) > 2 and group[2]:
                train['fine_validation_each_nth_epoch'] = n = group[2]
                assert n >= 0, 'fine_validation_each_nth_epoch < 0'

        elif group.startswith('max='):
            train['max_epoch'] = n = int(group[4:])
            assert n >= 0, 'max_epoch < 0'

        elif group.startswith('update='):
            train['update_every_n_batch'] = n = int(group[7:])
            assert n > 0, 'update_every_n_batch <= 0'
    
        elif group == '!':
            train['test_with_validation'] = True

        elif group == 'mp':
            train['multiprocessing_decode'] = True

        elif group.startswith('optuna='):
            train['optuna_trials'] = n = int(group[7:])
            assert n >= 0, 'optuna_trials < 0'

        elif group:
            raise ValueError('Unknown training param:' + group)

    from math import ceil
    (max_epoch, fine_validation_at_nth_wander, stop_at_nth_wander,
     fine_validation_each_nth_epoch) = (train[x] for x in 'max_epoch fine_validation_at_nth_wander stop_at_nth_wander fine_validation_each_nth_epoch'.split())
    expected_min_epoch = fine_validation_at_nth_wander + ceil(stop_at_nth_wander / fine_validation_each_nth_epoch)
    if max_epoch < expected_min_epoch:
        from utils.shell_io import byte_style
        from sys import stderr
        print(byte_style(f'[WARNING] max_epoch({max_epoch}) < expected_min_epoch({expected_min_epoch})', '3'), file = stderr)

    return train