from utils.str_ops import strange_to

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
    op_code = instance and instance[0].isalpha() and instance[0]
    exp_ids = instance[1:] if op_code else instance
    exp_ids = strange_to(exp_ids, str) if exp_ids else [exp_ids]
    return op_code, exp_ids

import re
def check_train(train_str):
    # >4/4|30:100!
    train = {}
    match = re.match(r'>(\d+)', train_str)
    train['fine_validation_at_nth_wander'] = int(match.group(1)) if match else 5

    match = re.match(r'/(\d+)', train_str)
    train['fine_validation_each_nth_epoch'] = int(match.group(1)) if match else 4

    match = re.match(r'\|(\d+)', train_str)
    train['stop_at_nth_wander'] = int(match.group(1)) if match else 100

    match = re.match(r':(\d+)', train_str)
    train['max_epoch'] = int(match.group(1)) if match else 200

    match = re.match(r'\^(\d+)', train_str)
    train['test_with_validation'] = bool('!' in train_str)

    return train