import yaml
from os import remove
from os.path import isfile
from datetime import datetime
from utils.shell_io import byte_style

def _lock_msg(lfile):
    if isfile(lfile):
        head = 'Wait for ' + byte_style(f"'{lfile}'", '3')
        content = '(Locked at '
        with open(lfile) as fr:
            for line in fr:
                content += line.split('.')[0]
        return head, content.strip() + ')'

def _wait_or_exit(lfile):
    if msg := _lock_msg(lfile):
        msg = '\n '.join(msg)
        msg += ' Unlock? [Y or any key to exit] '
        if input(msg) != 'Y':
            # print('\nexited')
            exit()
        if isfile(lfile):
            remove(lfile)

def _wait(lfile):
    from time import sleep
    second = 0
    while msg := _lock_msg(lfile):
        head, content = msg
        if second == 0:
            print(head)
        print('\r  ' + content + f' {second} second', end = '')
        sleep(1)
        second += 1
    print()


def _block(lfile):
    with open(lfile, 'w') as fw:
        fw.write(str(datetime.now()))

def _unblock(lfile):
    if isfile(lfile):
        remove(lfile)
        return True
    return False

def save_yaml(status, mfile, lfile, wait_lock = True):
    if lfile:
        if wait_lock:
            _wait(lfile)
            _block(lfile)

        finished = False
        do_exit = None
        while not finished:
            try:
                with open(mfile, 'w') as fw:
                    fw.write(f'# {datetime.now()}\n')
                    yaml.safe_dump(status, fw, encoding = 'utf-8', allow_unicode = True) #default_flow_style = False
                finished = True
            except KeyboardInterrupt as e:
                do_exit = e
                print('suppress', e, 'for saving', mfile)
        # release
        if wait_lock:
            _unblock(lfile)
            if do_exit is not None:
                raise do_exit
    else:
        with open(mfile, 'a+') as fw:
            fw.write(f'# {datetime.now()}\n')
            yaml.dump(status, fw, default_flow_style = False)
    return True

def load_yaml(mfile, lfile, wait = True, wait_then_block = False, wait_or_exit = True):
    if isfile(mfile):
        if wait:
            if wait_or_exit:
                _wait_or_exit(lfile)
            else:
                _wait(lfile)
            if wait_then_block:
                _block(lfile)
    else:
        save_yaml({}, mfile, lfile)
    with open(mfile, 'r') as fr:
        status = yaml.load(fr, Loader = yaml.FullLoader)
    if not status:
        status = {}
    if wait and wait_then_block:
        return status, lambda : _unblock(lfile)
    return status