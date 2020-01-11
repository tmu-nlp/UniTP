from subprocess import run, PIPE, Popen
from os.path import join

C_RED = '\033[31m' # Red Text
C_GREEN = '\033[32m' # Green Text
C_YELLOW = '\033[33m' # Yellow Text
C_BLUE = '\033[34m' # Blue Text
C_END = '\033[m' # reset to the defaults

red    = lambda x: C_RED + x + C_END
green  = lambda x: C_GREEN + x + C_END
yellow = lambda x: C_YELLOW + x + C_END
blue   = lambda x: C_BLUE + x + C_END

def call_fasttext(fasttext, wfile, vfile, ft_bin, ft_lower): # TODO: async
    # import pdb; pdb.set_trace()
    src = Popen(['cat', wfile], stdout = PIPE)
    src = Popen(['cut', '-f1'], stdin = src.stdout, stdout = PIPE)
    if ft_lower:
        src = Popen(['tr', '[:upper:]', '[:lower:]'], stdin = src.stdout, stdout = PIPE)
    dst = Popen([fasttext, 'print-word-vectors', ft_bin], stdin = src.stdout, stdout = PIPE)
    with open(vfile, 'wb') as fw: # TODO: save it in numpy format!
        dst = Popen(['cut', '-d ', '-f2-'], stdin = dst.stdout, stdout = fw) # space is intersting

def parseval(cmd_tuple, fhead, fdata):
    command = list(cmd_tuple)
    command.append(fhead)
    command.append(fdata)
    return run(command, stdout = PIPE, stderr = PIPE)

def rpt_summary(rpt_lines, get_individual, get_summary):
    individuals = get_individual
    summary = {}
    for line in rpt_lines.split('\n'):
        if line.startswith('===='):
            if individuals is True: # start
                individuals = []
            elif isinstance(individuals, list): # end
                individuals = tuple(individuals)
                if not get_summary:
                    return individuals
        elif isinstance(individuals, list):
            sent = tuple(float(s) if '.' in s else int(s) for s in line.split())
            individuals.append(sent)
        elif get_summary:
            if line.startswith('Number of sentence'):
                summary['N'] = int(line[line.rfind(' '):])
            if line.startswith('Bracketing Recall'):
                summary['LR'] = float(line[line.rfind(' '):])
            if line.startswith('Bracketing Precision'):
                summary['LP'] = float(line[line.rfind(' '):])
            if line.startswith('Bracketing FMeasure'):
                summary['F1'] = float(line[line.rfind(' '):])
            if line.startswith('Tagging accuracy'):
                summary['TA'] = float(line[line.rfind(' '):])
                break
            # ID  Len.  Stat. Recal  Prec.  Bracket gold test Bracket Words  Tags Accrac
    if get_individual:
        return individuals, summary
    return summary