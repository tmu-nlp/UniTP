from os import remove, mkdir, rmdir, walk, listdir, rename, link
from os.path import join, isdir, isfile, abspath, sep, basename, dirname
from shutil import copy
from array import array

def parpath(path, n = 1):
    path = abspath(path)
    par = -1
    for _ in range(n):
        par = path[:par].rindex(sep)
    return path[:par] if n else path

def path_folder(fpath):
    if fpath[-1] != sep:
        fpath += sep
    return fpath, fpath[:-1].split(sep)

def create_join(base_path, *succ, **kwargs):
    clear = kwargs.get('clear', False)
    for s in succ:
        base_path = join(base_path, s)
        if isdir(base_path):
            continue
        mkdir(base_path)
        clear = False
    if clear:
        rm_rf(base_path, kwargs.get('print_to', None))
        mkdir(base_path)
    return base_path

def rm_rf(fpath, print_to):
    for path, _, fnames in walk(fpath, False):
        if print_to:
            print(f'clear {len(fnames)} files in ' + path, file = print_to)
        for fname in fnames:
            remove(join(path, fname))
        rmdir(path)

def count_lines(fname, count_sep = False, sep = b'\n', buffer_size = 1 << 24):
    count = 0 # if count_empty_line else 1 # for the final line, like 5 fingers & 4 spaces
    with open(fname, 'rb') as fr:
        while True:
            buffer = fr.read(buffer_size)
            if not buffer: break
            if count_sep:
                count += buffer.count(sep)
            else:
                count += sum(1 for l in buffer.split(sep) if len(l))
    return count

def read_data(fname, v2i, lengths_and_lines = False, qbar = None):
    if v2i:
        byte = 1
        v_size, v2i = v2i
        v_size >>= 8
        while v_size:
            byte += 1
            v_size >>= 8
        if byte == 1:
            byte = 'B'
        elif byte == 2:
            byte = 'H'
        elif byte <= 4:
            byte = 'L'
        elif byte <= 8:
            byte = 'Q'
        else:
            raise ValueError(f'Too long: {v_size}')
    values = []
    if lengths_and_lines:
        lengths = []
        lines = []
    with open(fname) as fr:
        for line_id, line in enumerate(fr):
            words = line.rstrip().split()
            if v2i is None:
                value = words
            else:
                value = array(byte, (v2i(tok) for tok in words))
            values.append(value)
            if lengths_and_lines:
                lengths.append(len(value))
                lines.append(words)
            if qbar: qbar.update(1)
    if lengths_and_lines:
        return values, lengths, lines
    return values

def copy_with_prefix_and_rename(path_prefix, dst_path, new_prefix):
    if path_prefix.endswith(sep) or sep not in path_prefix:
        raise ValueError(f"full path '{path_prefix}' should include a prefix")
    src_path = dirname(path_prefix)
    f_prefix = basename(path_prefix)
    for mf in listdir(src_path):
        if mf.startswith(f_prefix):
            copy(join(src_path, mf), join(dst_path, f'{new_prefix + mf[len(f_prefix):]}'))

def transcode(in_file, out_file, enc, dec = 'utf-8'):
    with open(in_file, 'rb') as fr, open(out_file, 'wb') as fw:
         for line in fr:
             fw.write(line.decode(enc).encode(dec))

from signal import signal, SIGINT
from traceback import print_stack
from collections import defaultdict
from sys import stderr

class DelayedKeyboardInterrupt:
    def __init__(self, ignore = True, file = stderr):
        self._ignore = ignore
        self._file = file

    def __enter__(self):
        self.received_signals = defaultdict(lambda:defaultdict(int))
        self.old_handler = signal(SIGINT, self.handler)

    def __exit__(self, exception_type, exception_value, traceback):
        # swap back with the old handler
        signal(SIGINT, self.old_handler)
        for sig, frames in self.received_signals.items():
            sig = f'SIG#{sig} with {sum(frames.values())} '
            sig += 'recieval(s) from stack frame(s):'
            print(sig, file = self._file)
            for frame in frames:
                print_stack(frame, file = self._file)

    def handler(self, sig, frame):
        if self._ignore:
            return
        self.received_signals[sig][frame] += 1
        print('SIGINT (KeyboardInterrupt) received and delayed.', file = self._file)