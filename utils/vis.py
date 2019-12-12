from multiprocessing import Process, Queue
from time import sleep

class Vis(Process):

    def __init__(self, epoch):
        self._epoch = epoch
        self._q = Queue()

    def append(self, batch_id, batch, final = False):
        self._q.put((batch_id, batch, final))

    def run(self):
        self._before()
        while True:
            if self._q.empty():
                sleep(0.01)
                continue
            batch_id, batch, final = self._q.get()
            self._process(batch_id, batch)
            if final:
                break
        self._result = self._after()
        
    def _before(self):
        raise NotImplementedError()

    def _process(self, batch_id, batch):
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

    @property
    def result(self):
        return self._result

    @property
    def epoch(self):
        return self._epoch