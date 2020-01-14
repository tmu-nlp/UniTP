from multiprocessing import Process, Queue
from time import sleep

class VisWorker(Process): # designed for decoding batch
    def __init__(self, vis, in_q, out_q):
        super().__init__()
        self._vio = vis, in_q, out_q

    def run(self):
        vis, in_q, out_q = self._vio
        vis._before()
        while True:
            if in_q.empty():
                sleep(0.00001)
                continue
            inp = in_q.get()
            if inp is None:
                break
            args, kw_args = inp
            vis._process(*args, **kw_args)
        args, kw_args = in_q.get()
        outs = vis._after(*args, **kw_args)
        out_q.put((outs, vis._attrs))

class BaseVis:
    def __init__(self, epoch):
        self._epoch = epoch
        self._attrs = {}
        
    def _before(self):
        raise NotImplementedError()

    def _process(self, *args, **kw_args):
        raise NotImplementedError()

    def _after(self, *args, **kw_args):
        raise NotImplementedError()

    @property
    def epoch(self):
        return self._epoch

    def register_property(self, attr_name, value):
        self._attrs[attr_name] = value

    def __getattr__(self, attr_name):
        return self._attrs[attr_name]

class VisRunner:
    def __init__(self, vis, async_ = False):
        self._vis   = vis
        self._async = async_
        
    def before(self):
        if self._async:
            iq, oq = Queue(), Queue()
            worker = VisWorker(self._vis, iq, oq)
            self._async = worker, iq, oq
            worker.start()
        else:
            self._vis._before()

    def process(self, *args, **kw_args):
        if self._async:
            _, iq, _ = self._async
            iq.put((args, kw_args))
        else:
            self._vis._process(*args, **kw_args)

    def after(self, *args, **kw_args):
        if self._async:
            worker, iq, oq = self._async
            iq.put(None) # end while loop
            iq.put((args, kw_args)) # call _after
            worker.join()
            out, attrs = oq.get()
            self._vis._attrs.update(attrs)
            return out
        return self._vis._after(*args, **kw_args)

    def __getattr__(self, attr_name):
        return getattr(self._vis, attr_name)