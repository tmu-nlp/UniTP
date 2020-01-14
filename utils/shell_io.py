from subprocess import run, PIPE, Popen
from os.path import join

def byte_style(content, fg_color = '7', bg_color = '0', bold = False, dim = False, negative = False, underlined = False, blink = False):
    prefix = '\033['
    if not (bold or dim or negative or underlined or blink):
        prefix += '0;'
    if bold:
        prefix += '1;'
    if dim:
        prefix += '2;'
    if negative:
        prefix += '3;'
    if underlined:
        prefix += '4;'
    if blink:
        prefix += '5;'

    prefix += f'3{fg_color};4{bg_color}m'
    return prefix + content + '\033[0;37;40m'

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