C_RED = '\033[31m' # Red Text
C_GREEN = '\033[32m' # Green Text
C_YELLOW = '\033[33m' # Yellow Text
C_END = '\033[m' # reset to the defaults

from collections import defaultdict
def default_components(*d_comp):
    return defaultdict(lambda: d_comp) # just tuple

import numpy as np
def list_combine_add(new_info, old_info, extra = None):
    if old_info is None:
        if isinstance(new_info, (list, dict)):
            return new_info
        # convert tuple into list
        assert isinstance(new_info, tuple)
        return list(new_info)
    elif not hasattr(new_info, '__iter__'):
        # basic elements that can be added
        return new_info + old_info

    if isinstance(new_info, dict):
        for k, v in new_info.items():
            if isinstance(v, (int, float, np.ndarray)): # Counter
                old_info[k] += v
            else:
                old_info[k] = list_combine_add(v, old_info[k])
    else:
        for i, n in enumerate(new_info):
            if isinstance(n, (tuple, list)):
                old_info[i] = list_combine_add(n, old_info[i])
            else:
                old_info[i] += n
    return old_info

from utils.math_ops import harmony
def harmony_all_fractions(results):
    fractions = []
    for scores in results.values():
        if not isinstance(scores, dict):
            continue
        for v in scores.values():
            if isinstance(v, float):
                if 0 < v <= 1:
                    fractions.append(v)
                if v == 0:
                    fractions.append(1e-10)
    return harmony(fractions)

from tqdm import tqdm
from utils.types import M_TRAIN, M_DEVEL, M_TEST
def run(epoch_components,    # JT: {defaultdict_epoch: ('data_root', ....)}
        train_op,            # JT: {'data_root': train_op, ...}
        train_num_node,      # JT: {'data_root': train_num_node, ...}
        train_log_callback,  # JT: func('data_root', z, list)
        train_list,          # JT: {'data_root': train_list, ....} # eval_key_func
        eval_list_func,      # JT: func('data_root', eMajor, eMinor)
        eval_result_callback,# JT: func('data_root', z, list)
        d,                   # JT: {mode:'data_root':batch}
        recorder             = None,
        eval_key_func        = harmony_all_fractions, # JT: sum up joint task results
        train_feed_dict      = {}, # JT
        train_list_combine   = {}, # JT
        eval_list_combine    = {}, # JT
        max_epoch            = 100,
        print_each_n_batch   = 50,
        stop_at_nth_wander   = 20,
        mute_eval            = 30,
        fine_eval_at_nth_wander  = 3,
        fine_eval_each_nth_epoch = 5,
        test_with_eval       = True,
        **test_kwargs):

    def eval_betterment():
        time_label = f'{epoch_cnt}-{eval_cnt}'
        results = {'tid': time_label}
        num_sample = tuple(size for n, size in d[M_DEVEL].items() if n.endswith('_size'))
        desc = f'Eval {time_label}'
        if len(num_sample) > 1:
            desc += f' [{"+".join(str(s) for s in num_sample)}]'
        with tqdm(total = sum(num_sample), desc = desc) as qbar:
            desc = []
            for data_root, batch_spec in d[M_DEVEL].items():
                cumu_el = None
                cache = {}
                for batch in batch_spec.iter:
                    el, nb = sess.run([eval_list_func[data_root](time_label), eval_batch_size[data_root]])
                    cumu_el = eval_list_combine.get(data_root, list_combine_add)(el, cumu_el, (bid, epoch_cnt, eval_cnt, recorder, cache))
                    qbar.update(nb)
                inner, outer, scores = eval_result_callback[data_root](bid, cumu_el, epoch_cnt, eval_cnt, cache)
                desc.append((bid, data_root.title().replace('_B', ' b'), inner, outer))
                results[data_root] = scores
            results['key'] = key = eval_key_func(results)
            betterment = recorder.check_betterment(time_label, results)
            line = 'Eval '
            for num, name, inner, outer in desc:
                recorder(f'Eval {num} {name} {inner} {outer}')
                line += f'{outer}; ' if len(desc) > 1 else  f'{num} {name} {outer}; '
            qbar.desc = (C_GREEN if betterment else C_YELLOW) + line + f"‹{key:.2f}›" + C_END
            recorder(f'Eval betterment, wander({nth_wander}|{stop_at_nth_wander}) ‹{key:.2f}›')
        return betterment
    
    train_batch_size = {data_root: d[M_TRAIN][data_root].size for data_root in train_op}
    eval_batch_size  = {data_root: d[M_DEVEL ][data_root].size for data_root in train_op}
    with tf.name_scope('train_log'):
        vname_shapes = {v.name: tf.shape(v) for v in tf.trainable_variables()}

    with tf.Session(config = config) as sess:
        epoch_cnt, eval_cnt = recorder.initial_or_restore(sess)
        eval_nth  = 1/fine_eval_each_nth_epoch
        nth_wander = 0
        start_fine_eval = nth_wander >= fine_eval_at_nth_wander
    
        if epoch_cnt + eval_cnt > 0:
            max_epoch += epoch_cnt
            sess.run(tf.tables_initializer())
        else:
            _, vname_shapes = sess.run([tf.tables_initializer(), vname_shapes]) # for datasets
            num_variables = 0
            _ = ''
            for vname in sorted(vname_shapes):
                shape = vname_shapes[vname]
                size = np.prod(shape)
                num_variables += size
                _ += '\n' + f'{size:,d}'.rjust(15) 
                _ += ('(' + ':'.join(str(i) for i in shape)).rjust(15) + ')\t'
                _ += vname
            recorder(f'Total numbers of parameters: {num_variables:,d}' + _)

        while True: # for epoch
            num_sample = 0
            component_cnt   = defaultdict(int)
            component_lives = defaultdict(list)
            for data_root in epoch_components[epoch_cnt]:
                n = d[M_TRAIN][data_root].num_sample
                num_sample               += n
                component_cnt[data_root] += n
                component_lives[data_root].append(d[M_TRAIN][data_root].initializer)
            components, component_vec = zip(*component_cnt.items())
            component_vec = list(component_vec)

            sampl_cnt = {}
            batch_cnt = {}
            token_cnt = {} # track speed of the training process
            cumu_list = {}
            for data_root in components:
                sampl_cnt[data_root] = 0
                batch_cnt[data_root] = 0
                token_cnt[data_root] = 0
                cumu_list[data_root] = None
                sess.run(component_lives[data_root].pop())
                d[M_TRAIN][data_root].reinit_rand()
            start_time = time()
            finish_epoch = False
            comp_i = 0
            comp_n = len(components)
            data_root = '----'
            with tqdm(desc = f'Epoch {epoch_cnt} wander({nth_wander}|{stop_at_nth_wander})', total = num_sample) as qbar:
                while True: # num_sample - sum(sampl_cnt.values())
                    if comp_n > 1: # JT+
                        remains = sum(component_vec)
                        desc = f'Epoch {epoch_cnt} {data_root[:4].title()}»({",".join(str(sampl_cnt[d]) for d in components)})'
                        if remains:
                            prob = tuple(c/remains for c in component_vec)
                            desc += f'[{" ".join(str_percentage(p, "%.1f") for p in prob)}]'
                            comp_i = np.random.choice(comp_n, p = prob)
                        else:
                            desc += '[finish JT]'
                            comp_i = 0
                        qbar.desc = desc + f' wander({nth_wander}|{stop_at_nth_wander})'
                    data_root = components[comp_i]
                    percentage = sum(sampl_cnt.values()) / num_sample
                    e_time = epoch_cnt + percentage
                    try:
                        # train
                        _, n, tl, nb, sm = sess.run([train_op        [data_root],
                                                     train_num_node  [data_root],
                                                     train_list      [data_root],
                                                     train_batch_size[data_root],
                                                     recorder.summary_op()],
                                                    feed_dict = train_feed_dict.get(data_root, default_feed_dict)(e_time, (nth_wander / stop_at_nth_wander + percentage % (1/fine_eval_each_nth_epoch))))
                        recorder.write_summary(sm) # e_time
                        sampl_cnt[data_root] += nb
                        batch_cnt[data_root] += 1
                        token_cnt[data_root] += n
                        component_vec[comp_i] -= nb
                        cumu_list[data_root] = train_list_combine.get(data_root, list_combine_add)(tl, cumu_list[data_root])
                        qbar.update(nb)
                        # if np.isnan(cumu_list[data_root][-1]):
                        #     print('nan!', batch_cnt[data_root], n)
                        if batch_cnt[data_root] % print_each_n_batch == 0:
                            s = 'Train Epoch %.2f' % e_time
                            s += ' %6d nps.' % (token_cnt[data_root] / (time() - start_time))
                            s += train_log_callback[data_root](print_each_n_batch, cumu_list[data_root])
                            recorder(s)
                            start_time = time()
                            token_cnt[data_root] = 0
                            cumu_list[data_root] = None
                    except tf.errors.OutOfRangeError:
                        if sum(len(l) for l in component_lives.values()) == 0:
                            finish_epoch = True
                        else: # just for JT
                            recorder(f"reinit {data_root} during a epoch")
                            sess.run(component_lives[data_root].pop())
                            
                    if percentage >= eval_nth and start_fine_eval or finish_epoch and mute_eval == 0:
                        start_time = time() - start_time
                        nth_wander = 0 if eval_betterment() else nth_wander + 1
                        qbar.desc = f'Epoch {epoch_cnt} wander({nth_wander}|{stop_at_nth_wander})'
                        if nth_wander >= fine_eval_at_nth_wander:
                            start_fine_eval = True
                        if nth_wander >= stop_at_nth_wander:
                            qbar.desc = C_RED + f'Stopping training ... ({nth_wander}|{stop_at_nth_wander})' + C_END
                            scores = recorder.test_model(sess, d, **test_kwargs)
                            # tf.reset_default_graph()
                            return scores
                        elif test_with_eval:
                            recorder.test_model(sess, d, **test_kwargs, current_epoch_major_monir = (epoch_cnt, eval_cnt))
                        eval_nth += 1/fine_eval_each_nth_epoch
                        eval_cnt += 1
                        start_time = time() - start_time
                    if mute_eval > 0:
                        mute_eval -= 1
                    if finish_epoch:
                        break
            epoch_cnt += 1
            eval_nth = 1/fine_eval_each_nth_epoch
            eval_cnt = 0
            if epoch_cnt == max_epoch:
                break
        scores = recorder.test_model(sess, d, **test_kwargs)
    return scores

def run_little(recorder,
               d,
               test_result_callback,
               test_list_func       = {}, # JT
               test_list_combine    = {}, # JT
               ):
    with tf.Session(config = config) as sess:
        sess.run(tf.tables_initializer()) # for datasets
        score = recorder.test_model(sess,
                                    d,
                                    test_result_callback,
                                    test_list_func,
                                    test_list_combine)
    # tf.reset_default_graph()
    return score