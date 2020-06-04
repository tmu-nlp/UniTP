# from utils.param_ops import iter_zipped_nt_params, zip_nt_params, change_key, dict_print
# from utils.yaml_io import load_yaml, save_yaml
# from utils.file_io import create_join, join, isfile, isdir, listdir, rm_rf, copy_with_prefix_and_rename
from utils.param_ops import HParams

def train(train_params, operator):
    train_params = HParams(train_params)
    epoch_cnt, fine_validation = operator.train_initials()
    nth_wander = train_params.fine_validation_at_nth_wander if fine_validation else 0
    validation_each_nth_epoch = train_params.fine_validation_each_nth_epoch
    # TODO: set fine_validation_at_nth_wander to
    for epoch_cnt in range(epoch_cnt, train_params.max_epoch):
        nth_validation = 1 if fine_validation else train_params.fine_validation_each_nth_epoch
        train_step = operator.train_step(epoch_cnt, nth_wander / train_params.stop_at_nth_wander)
        for percentage in train_step:
            if percentage >= (nth_validation / validation_each_nth_epoch):
                epoch = epoch_cnt + nth_validation / validation_each_nth_epoch
                if operator.validate_betterment(epoch, nth_wander == train_params.fine_validation_at_nth_wander):
                    nth_wander = 0
                else:
                    nth_wander += 1
                if percentage < 1:
                    train_step.send(nth_wander / train_params.stop_at_nth_wander) # schedule
                if nth_wander > train_params.fine_validation_at_nth_wander:
                    fine_validation = True
                    if nth_wander >= train_params.stop_at_nth_wander:
                        return operator.test_model()
                if train_params.test_with_validation:
                    operator.test_model(epoch = epoch)
                nth_validation += 1
        nth_validation = validation_each_nth_epoch + 1
    return operator.test_model()