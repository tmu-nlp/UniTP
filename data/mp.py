from multiprocessing import Pool, Process, Queue, TimeoutError
from math import ceil
from time import time, sleep

from sys import stderr
from os.path import join, dirname
from utils.shell_io import concatenate, byte_style
from utils.file_io import join, create_join

t_sleep_deep = 0.1
t_sleep_shallow = 0.001
t_awake = 0.00001

def mp_desc(final, *largs):
    prefix = '['
    if final:
        tree_count, total_time, error_count = largs
        suffix = '] ' + byte_style(f'✔ {tree_count} trees', '2')
        if error_count:
            suffix += byte_style(f' (✗ {error_count})', '1')
        suffix += byte_style(f' ({total_time:.1f} sec.)', '2')
    else:
        finished_count, total = largs
        if finished_count:
            suffix = f'] {finished_count}/{total} ended'
        else:
            suffix = f'] {total} threads'
    return prefix, suffix

def mp_while(worker_cls, jobs, receive_fn, *args, desc_fn = mp_desc):
    from utils.str_ops import StringProgressBar
    from utils.file_io import DelayedKeyboardInterrupt
    from data.io import distribute_jobs
    from utils.types import num_threads
    q = Queue()
    num_threads = min(num_threads, len(jobs))
    jobs = distribute_jobs(jobs, num_threads)
    tree_count = error_count = thread_join_count = 0
    prefix, suffix = desc_fn(False, thread_join_count, num_threads) if callable(desc_fn) else '[]'
    start_time = time()
    with StringProgressBar.segs(num_threads, prefix = prefix, suffix = suffix) as qbar:
        try:
            for i in range(num_threads):
                if not worker_cls.estimate_total:
                    qbar.update(i, total = len(jobs[i]))
                jobs[i] = w = worker_cls(i, q, jobs[i], *args)
                w.start()
            while True:
                if q.empty():
                    sleep(t_awake)
                elif (status := receive_fn(q.get(), qbar)) is not None:
                    i, tree_cnt, error_cnt = status
                    error_count += error_cnt
                    jobs[i].join()
                    thread_join_count += 1
                    tree_count += tree_cnt
                    if callable(desc_fn):
                        qbar.desc = desc_fn(False, thread_join_count, num_threads)
                    if thread_join_count == num_threads:
                        break
        except (KeyboardInterrupt, Exception) as ex:
            with DelayedKeyboardInterrupt():
                for x in jobs:
                    x.kill()
            raise ex
        if callable(desc_fn):
            qbar.desc = desc_fn(True, tree_count, time() - start_time, error_count)
    return tree_count


class D2T(Process):
    def __init__(self, idx, in_q, out_q, vocabs, tree_gen_fn, cat_dir):
        super().__init__()
        self._id_q_vocabs_fn = idx, in_q, out_q, vocabs, tree_gen_fn, cat_dir

    def run(self):
        idx, in_q, out_q, vocabs, tree_gen_fn, cat_dir = self._id_q_vocabs_fn
        t_sleep = t_sleep_shallow
        last_wake = time()
        while True:
            while in_q.empty():
                sleep(t_sleep)
                if time() - last_wake > 5:
                    t_sleep = t_sleep_deep
                else:
                    t_sleep = t_sleep_shallow
                continue
            signal = in_q.get()
            last_wake = time()
            if signal is None:
                out_q.put(idx)
                continue
            elif isinstance(signal, int):
                if signal < 0:
                    break
            key, tensor_args, corp_key = signal
            if isinstance(vocabs, dict):
                i2vs_args = vocabs[corp_key] + tensor_args
            else:
                i2vs_args = vocabs + tensor_args
            tree_gen = tree_gen_fn(*i2vs_args)
            if cat_dir:
                fname = join(cat_dir, 'mp.%d_%d.tree' % key)
                with open(fname, 'w') as fw:
                    fw.write('\n'.join(tree_gen))
                out_q.put((key, fname))
            else:
                out_q.put((key, list(tree_gen)))
            last_wake = time()
            

class DM:
    @staticmethod
    def tree_gen_fn():
        raise NotImplementedError()

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        raise NotImplementedError()

    def __init__(self, batch_size, vocabs, num_workers, fdata = None, cat_files = False):
        rin_q = Queue()
        rout_q = Queue()
        self._q_receiver = rin_q, rout_q, None
        fpath = dirname(fdata) if fdata and cat_files else None
        q_workers = []
        for seg_id in range(num_workers):
            in_q = Queue()
            d2t = D2T(seg_id, in_q, rin_q, vocabs, self.tree_gen_fn, fpath)
            d2t.start()
            q_workers.append((in_q, d2t))
        self._mp_workers = q_workers, ceil(batch_size / num_workers), batch_size, fdata, cat_files
        self._timer = time()

    def timeit(self):
        self._timer = time()

    def batch(self, batch_id, *args, key = None): # split a batch for many workers
        q_workers, seg_size, batch_size, fdata, cat_files = self._mp_workers
        rin_q, rout_q, tr = self._q_receiver
        if tr is None:
            tr = TR(rin_q, rout_q, [False for _ in q_workers], fdata, cat_files)
            tr.start()
            self._q_receiver = rin_q, rout_q, tr
            
        for seg_id, (in_q, _) in enumerate(q_workers):
            major_args = self.arg_segment_fn(seg_id, seg_size, batch_size, args)
            if major_args:
                in_q.put(((batch_id, seg_id), major_args, key))

    def batched(self):
        q_workers, _, _, fdata, _ = self._mp_workers
        for in_q, _ in q_workers:
            in_q.put(None)
        rin_q, rout_q, tr = self._q_receiver
        trees_time = rout_q.get()
        if fdata:
            trees = None
        else:
            trees, trees_time = trees_time # trees
        self._timer = trees_time - self._timer
        tr.join()
        self._q_receiver = rin_q, rout_q, None
        return trees

    @property
    def duration(self):
        return self._timer

    def close(self):
        q_workers, _, _, _, _ = self._mp_workers
        good_end = bad_end = 0
        for in_q, _ in q_workers:
            in_q.put(-1) # let them response
        for in_q, d2t in q_workers:
            try:
                d2t.join(timeout = 0.5)
                good_end += 1
            except TimeoutError:
                bad_end += 1
            d2t.terminate()
        _, _, tr = self._q_receiver
        force_tr = False
        if tr is not None:
            try:
                tr.join(timeout = 0.5)
                good_end += 1
            except TimeoutError:
                bad_end += 1
                force_tr = True
            tr.terminate()
        good_end = byte_style(str(good_end), '2')
        if bad_end:
            bad_end = str(bad_end)
            if force_tr:
                bad_end += ' (TR)'
            bad_end = byte_style(bad_end, '1')
            bad_end = f'Terminate {bad_end} processes by force; '
        else:
            bad_end = ''
        print(bad_end + f'{good_end} processes naturally ended.', file = stderr)

    def __str__(self):
        q_workers, seg_size, batch_size, fdata, cat_files = self._mp_workers
        line = self.__class__.__name__ + f' ({id(self):X})'
        line += f' with {len(q_workers)} workers for seg_size ({seg_size}) of batch_size ({batch_size})'
        if fdata:
            line += ' to \'' + fdata + '\''
        return line


class TR(Process):
    def __init__(self, in_q, out_q, checklist, fdata, cat_files, flatten_batch = True):
        super().__init__()
        self._q = in_q, out_q, checklist, fdata, cat_files, flatten_batch

    def run(self):
        i_trees = {}
        in_q, out_q, checklist, fdata, cat_files, flatten_batch = self._q
        t_sleep = t_sleep_shallow
        last_wake = time()
        while True:
            while in_q.empty():
                sleep(t_sleep)
                if time() - last_wake > 5:
                    t_sleep = t_sleep_deep
                else:
                    t_sleep = t_sleep_shallow
                continue
            signal = in_q.get()
            last_wake = time()
            if isinstance(signal, int):
                checklist[signal] = True # idx
                # print(checklist)
                if all(checklist):
                    break
                continue
            key, trees = signal
            i_trees[key] = trees
            last_wake = time()
        end_time = time()
            
        # print(sorted(i_trees))
        if fdata:
            if cat_files:
                cat_files = []
                for key in sorted(i_trees):
                    cat_files.append(i_trees[key])
                concatenate(cat_files, fdata)
            else:
                with open(fdata, 'w') as fw:
                    for key in sorted(i_trees):
                        fw.write('\n'.join(i_trees[key]) + '\n')
            out_q.put(end_time)
        else:
            trees = []
            for key in sorted(i_trees):
                bid, _ = key
                # print(key, len(i_trees[key]))
                if flatten_batch:
                    trees.extend(i_trees[key])
                elif bid == len(trees):
                    trees.append(i_trees[key])
                else:
                    trees[bid].extend(i_trees[key])
            trees = '\n'.join(trees) if flatten_batch else trees
            out_q.put((trees, end_time))


class BaseVis:
    def __init__(self, epoch, work_dir, i2vs):
        self._work_dir = work_dir
        self._epoch = epoch
        self._i2vs = i2vs

    @property
    def epoch(self):
        return self._epoch

    @property
    def i2vs(self):
        return self._i2vs

    def join(self, *fname):
        return join(self._work_dir, *fname)

    def create_folder(self, *fpath):
        return create_join(self._work_dir, fpath)
        
    def _before(self):
        pass

    def _process(self, *args, **kw_args):
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

class VisWorker(Process): # designed for decoding batch
    def __init__(self, vis, in_q, out_q):
        super().__init__()
        self._vio = vis, in_q, out_q

    def run(self):
        vis, in_q, out_q = self._vio
        vis._before()
        proc_time = 0
        while True:
            if in_q.empty():
                sleep(t_sleep_shallow)
                continue
            inp = in_q.get()
            if inp is None:
                break
            args, kw_args = inp
            start      = time()
            try:
                vis._process(*args, **kw_args)
            except Exception as err:
                if hasattr(vis, 'close'):
                    vis.close()
                raise err
            proc_time += time() - start
        out_q.put((vis._after(), proc_time))

class VisRunner:
    def __init__(self, vis, async_ = False):
        self._vis   = vis
        self._timer = 0
        
        if async_:
            iq, oq = Queue(), Queue()
            worker = VisWorker(self._vis, iq, oq)
            self._async = worker, iq, oq
            worker.start()
        else:
            self._async = None
            self._vis._before()
        self._duration = time()

    def process(self, *args, **kw_args):
        if self._async:
            _, iq, _ = self._async
            iq.put((args, kw_args))
        else:
            start = time()
            try:
                self._vis._process(*args, **kw_args)
            except Exception as err:
                if hasattr(self._vis, 'close'):
                    self._vis.close()
                raise err
            
            self._timer += time() - start

    def after(self):
        self._duration = time() - self._duration
        if self._async:
            worker, iq, oq = self._async
            iq.put(None) # end while loop | _after
            out, proc_time = oq.get()
            worker.join() # this should be after oq.get()
            self._timer = proc_time
            return out
        return self._vis._after()

    def __del__(self):
        if isinstance(self._async, tuple):
            self._async[0].terminate()

    @property
    def proc_time(self):
        return self._timer

    @property
    def before_after_duration(self):
        return self._duration

    @property
    def is_async(self):
        return isinstance(self._async, tuple)

    def __getattr__(self, attr_name):
        return getattr(self._vis, attr_name)