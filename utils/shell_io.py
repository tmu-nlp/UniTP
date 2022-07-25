from posixpath import join
from subprocess import run, PIPE, Popen

from utils.math_ops import inv_sigmoid

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
    return prefix + content + '\033[m'

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

from collections import Counter
def rpt_summary(rpt_lines, get_individual, get_summary):
    summary = {}
    individuals = get_individual
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

def concatenate(src_files, dst_file):
    command = ['cat']
    command.extend(src_files)
    rs = run(command, stdout = PIPE, stderr = PIPE)
    with open(dst_file, 'wb') as fw:
        fw.write(rs.stdout)
    assert not rs.stderr

def has_discodop():
    try:
        command = ['discodop', 'eval']
        run(command, stdout = PIPE, stderr = PIPE)
    except:
        return False
    return True

def discodop_tmp(hds, prm_file = None):
    from tempfile import TemporaryDirectory
    tmp = TemporaryDirectory()
    hf = join(tmp.name, 'head.export')
    df = join(tmp.name, 'data.export')
    with open(hf, 'w') as hw, open(df, 'w') as dw:
        for cnt, (head, data) in enumerate(hds):
            bos = f'#BOS {cnt + 1}\n'
            hw.write(bos)
            dw.write(bos)
            assert head[0].startswith('#BOS ')
            assert data[0].startswith('#BOS ')
            assert head[-1].startswith('#EOS ')
            assert data[-1].startswith('#EOS ')
            for hl, dl in zip(head[1:-1], data[1:-1]):
                hw.write(hl)
                dw.write(dl)
            eos = f'#EOS {cnt + 1}\n'
            hw.write(eos)
            dw.write(eos)
    return discodop_eval(hf, df, prm_file)

def discodop_eval(fhead, fdata, prm_file, rpt_file = None, get_individual = False):
    return discodop_smy(*discodop_eval_run(fhead, fdata, prm_file, rpt_file, get_individual), get_individual)

def discodop_eval_run(fhead, fdata, prm_file, rpt_file, get_individual):
    command = ['discodop', 'eval', fhead, fdata, prm_file]
    if get_individual: command.append('--verbose')
    dst = run(command, stdout = PIPE, stderr = PIPE)
    total = dst.stdout.decode()
    command.append('--disconly')
    dst = run(command, stdout = PIPE, stderr = PIPE)
    discontinuous = dst.stdout.decode()
    if rpt_file:
        rpt_file.write('\n═══════════════════════════════════════════════════════════════════════════════════════\n')
        rpt_file.write('Results from discodop eval: [Total]\n')
        rpt_file.write(total)
        rpt_file.write('\n [Discontinuous Only]\n')
        rpt_file.write(discontinuous)
    return total, discontinuous

def __disc_indiv(long_line, end):
    individual = []
    start = long_line.find('ID Length  Recall  Precis Bracket   gold   cand  Words  POS  Accur.')
    long_line = long_line[start:end].split('\n')[2:]
    for line in long_line:
        numbers = line.split()
        if len(numbers) != 10 or not numbers[0].isdigit():
            break
        any_0div = False
        for x, y in enumerate(numbers):
            if x in (2, 3, 9):
                if y == '0DIV!':
                    y = 'nan'
                    any_0div = True
                numbers[x] = float(y)
            else:
                numbers[x] = int(y)
        if any_0div:
            print(line)
        individual.append(numbers)
    assert all(v == 1 for v in Counter(k[0] for k in individual).values())
    return individual

def discodop_smy(total, discontinuous, get_individual):
    smy_string = total.rfind('_________ Summary _________')
    if get_individual:
        indiv = __disc_indiv(total, smy_string)
    smy_string = total[smy_string:].split('\n')
    smy = dict(TF = 0, TP = 0, TR = 0, DF = 0, DP = 0, DR = 0)
    for line in smy_string:
        if line.startswith('number of sentences:'):
            smy['TN'] = int(line.split()[-1])
        elif line.startswith('labeled recall:'):
            smy['TR'] = float(line.split()[-1])
        elif line.startswith('labeled precision:'):
            smy['TP'] = float(line.split()[-1])
        elif line.startswith('labeled f-measure:'):
            smy['TF'] = float(line.split()[-1])
    smy_string = discontinuous.rfind('_________ Summary _________')
    if get_individual:
        indiv = indiv, __disc_indiv(discontinuous, smy_string)
        assert len(indiv[0]) == len(indiv[1])
    smy_string = discontinuous[smy_string:].split('\n')
    for line in smy_string:
        if line.startswith('number of sentences:'):
            smy['DN'] = int(line.split()[-1])
        elif line.startswith('labeled recall:'):
            smy['DR'] = float(line.split()[-1])
        elif line.startswith('labeled precision:'):
            smy['DP'] = float(line.split()[-1])
        elif line.startswith('labeled f-measure:'):
            smy['DF'] = float(line.split()[-1])
    if get_individual:
        
        return (smy,) + indiv
    return smy