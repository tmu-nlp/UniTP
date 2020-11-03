# Authors: Wolfgang Maier <maierw@hhu.de>,
# Andreas van Cranenburgh <a.w.vancranenburgh@uva.nl>
# Version: January 24, 2014
# Modified By zchen0420@github, 2020

from collections import Counter, defaultdict
from utils.math_ops import bit_fanout
from utils.param_ops import HParams

def read_param(filename):
    validkeysonce = ('DEBUG', 'MAX_ERROR', 'CUTOFF_LEN', 'LABELED',
                     'DISC_ONLY', 'TED', 'DEP')
    param = {'DEBUG': 0, 'MAX_ERROR': 10, 'CUTOFF_LEN': 9999,
             'LABELED': 1, 'DELETE_LABEL_FOR_LENGTH': set(),
             'DELETE_LABEL': set(), 'DELETE_WORD': set(),
             'EQ_LABEL': set(), 'EQ_WORD': set(),
             'DISC_ONLY': 0, 'TED': 0, 'DEP': 0}
    seen = set()
    with open(filename) as fr:
        for line in fr:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, val = line.split(None, 1)
            if key in validkeysonce:
                if key in seen:
                    raise ValueError('cannot declare %s twice' % key)
                seen.add(key)
                param[key] = int(val)
            elif key in ('DELETE_LABEL', 'DELETE_LABEL_FOR_LENGTH',
                    'DELETE_WORD'):
                param[key].add(val)
            elif key in ('EQ_LABEL', 'EQ_WORD'):
                # these are given as undirected pairs (A, B), (B, C), ...
                try:
                    b, c = val.split()
                except ValueError:
                    raise ValueError('%s requires two values' % key)
                param[key].add((b, c))
            else:
                raise ValueError('unrecognized parameter key: %s' % key)
    return HParams(param)

def _scores(bracket_match, p_num_brackets, g_num_brackets):
    if p_num_brackets > 0:
        prec = 100 * bracket_match / p_num_brackets
    else:
        prec = 0.0
    if g_num_brackets > 0:
        rec = 100 * bracket_match / g_num_brackets
    else:
        rec = 0.0
    if prec + rec > 0:
        fb1 = 2 * prec * rec / (prec + rec)
    else:
        fb1 = 0.0
    return prec, rec, fb1

def incomplete_sent_line(disc_mark, sent_cnt, g_num_brackets, g_disc_num_brackets, g_tag_count, *p_tag_match_count):
    sent_line = f'╎ {disc_mark} {sent_cnt:5d} ╎                       ╎       {g_num_brackets:3d}       '
    if g_disc_num_brackets:
        sent_line += f'╎       {g_disc_num_brackets:3d}       '
    else:
        sent_line += f'╎                 '
    if p_tag_match_count:
        tag_match, p_tag_count = p_tag_match_count
        sent_line += f'╎  {tag_match:3d}   {p_tag_count:3d}  '
        sent_line += '   ╎' if g_tag_count == p_tag_count else ' x ╎'
    else:
        sent_line += f'╎          {g_tag_count:3d}     ╎'
    return sent_line

class DiscoEvalb:
    def __init__(self):
        self._tick = 0
        self._missing = 0
        self._total_match = 0
        self._total_pred = 0
        self._total_gold = 0
        self._total_exact = 0
        self._total_pos = 0
        self._total_matched_pos = 0
        self._total_sents = 0
        self._disc_sents = 0
        self._disc_match = 0
        self._disc_pred = 0
        self._disc_gold = 0
        self._disc_exact = 0
        self._sent_lines = []

    def add(self, p_brackets, p_tags, g_brackets, g_tags):
        self._total_sents += 1
        g_tag_count = len(g_tags)
        g_num_brackets = sum(g_brackets.values())
        g_disc_brackets = Counter({(lb, fo):ct for (lb, fo), ct in g_brackets.items() if bit_fanout(fo) > 1})
        g_disc_num_brackets = sum(g_disc_brackets.values())
        self._total_gold += g_num_brackets
        self._disc_gold += g_disc_num_brackets
        disc_mark = '* '[not g_disc_num_brackets]
        if g_disc_num_brackets:
            self._disc_sents += 1
        if p_tags is None:
            if p_brackets is None:
                self._sent_lines.append(incomplete_sent_line(disc_mark, self._total_sents - self._tick, g_num_brackets, g_disc_num_brackets, g_tag_count))
        else:
            tag_match = len(p_tags & g_tags)
            p_tag_count = len(p_tags)
            self._total_matched_pos += tag_match
            self._total_pos += g_tag_count
            if p_brackets is None:
                self._sent_lines.append(incomplete_sent_line(disc_mark, self._total_sents - self._tick, g_num_brackets, g_disc_num_brackets, g_tag_count, tag_match, p_tag_count))
        if p_brackets is None:
            self._missing += 1
            return -1, -1, -1, -1 if p_tags is None else (tag_match / p_tag_count)

        bracket_match = sum((g_brackets & p_brackets).values())
        p_num_brackets = sum(p_brackets.values())
        p_disc_brackets = Counter({(lb, fo):ct for (lb, fo), ct in p_brackets.items() if bit_fanout(fo) > 1})
        disc_bracket_match = sum((g_disc_brackets & p_disc_brackets).values())
        p_disc_num_brackets = sum(p_disc_brackets.values())
        if g_brackets == p_brackets:
            self._total_exact += 1
        if g_disc_num_brackets and g_disc_brackets == p_disc_brackets:
            self._disc_exact += 1
        sent_prec, sent_rec, sent_fb1 = _scores(bracket_match, p_num_brackets, g_num_brackets)
        sent_line =  f'╎ {disc_mark} {self._total_sents - self._tick:5d} ╎'
        sent_line += f' {sent_prec:6.2f} {sent_rec:6.2f}  {sent_fb1:6.2f} ╎'
        sent_line += f' {p_num_brackets:3d}   {bracket_match:3d}   {g_num_brackets:3d} ╎'
        if g_disc_num_brackets or p_disc_num_brackets:
            sent_line += f' {p_disc_num_brackets:3d}   {disc_bracket_match:3d}   {g_disc_num_brackets:3d} ╎'
        else:
            sent_line += '                 ╎'
        sent_line += f'  {tag_match:3d}   {p_tag_count:3d}  '
        sent_line += '   ╎' if g_tag_count == p_tag_count else ' x ╎'
        self._sent_lines.append(sent_line)
        self._total_match += bracket_match
        self._total_pred += p_num_brackets
        self._disc_match += disc_bracket_match
        self._disc_pred += p_disc_num_brackets
        return bracket_match, p_num_brackets, g_num_brackets, disc_bracket_match, p_disc_num_brackets, g_disc_num_brackets, tag_match, g_tag_count

    def add_batch_line(self, batch_id):
        batch_id = str(batch_id).rjust(4, '0')
        self._tick = self._total_sents
        self._sent_lines.append(f'├─ B{batch_id} ─┼───────────────────────┼─────────────────┼─────────────────┼────────────────┤')

    def __str__(self):
        tp, tr, tf, dp, dr, df = self.summary()
        line =  '╒═════════╤═════════════════════════════════════════╤═════════════════╤════════════════╕\n'
        line += '╎  Batch  ╎                     Total               ╎      Disco.     ╎        PoS     ╎\n'
        line += '╎ d sent. ╎  prec.   rec.    F1   | test match gold ╎ test match gold ╎ match gold err ╎\n'
        # line += '|──────────────────────────────────────────────────────────────────────────────────────|\n'
        line += '\n'.join(self._sent_lines) + '\n'
        line += '╘═════════╧═══════════════════════╧═════════════════╧═════════════════╧════════════════╛\n\n\n'
        line += 'Sentences in key'.ljust(30) + f': {self._total_sents}\n'
        line += 'Sentences missing in answer'.ljust(30) + f': {self._missing}\n'
        line += 'Total edges in key'.ljust(30) + f': {self._total_gold}\n'
        line += 'Total edges in answer'.ljust(30) + f': {self._total_pred}\n'
        line += 'Total matching edges'.ljust(30) + f': {self._total_match}\n\n'
        line += '    LP  : %6.2f %%\n' % tp
        line += '    LR  : %6.2f %%\n' % tr
        line += '    F1  : %6.2f %%\n' % tf
        line += '    EX  : %6.2f %%\n\n' % (100 * self._total_exact / self._total_sents if self._total_sents else 0)
        line += 'POS : %6.2f %%\n\n' % (100 * self._total_matched_pos / self._total_pos if self._total_pos else 0)
        line += 'Disc. Sentences in key'.ljust(30) + f': {self._disc_sents}\n'
        line += 'Total disc. edges in key'.ljust(30) + f': {self._disc_gold}\n'
        line += 'Total disc. edges in answer'.ljust(30) + f': {self._disc_pred}\n'
        line += 'Total matching disc. edges'.ljust(30) + f': {self._disc_match}\n\n'
        line += '    Disc. LP  : %6.2f %%\n' % dp
        line += '    Disc. LR  : %6.2f %%\n' % dr
        line += '    Disc. F1  : %6.2f %%\n' % df
        line += '    Disc. EX  : %6.2f %%\n' % (100 * self._disc_exact / self._disc_sents if self._disc_sents else 0)
        return line

    def summary(self):
        total = _scores(self._total_match, self._total_pred, self._total_gold)
        disc  = _scores( self._disc_match,  self._disc_pred,  self._disc_gold)
        return total + disc

    @property
    def total_missing(self):
        return self._total_sents, self._missing

def continuous_evalb(pred_fname, gold_fname, prm_fname):
    from data.cross import _read_dpenn, bracketing
    from nltk.tree import Tree
    evalb_lcfrs_prm = read_param(prm_fname)
    eq_w = evalb_lcfrs_prm.EQ_WORD
    eq_l = evalb_lcfrs_prm.EQ_LABEL
    args = dict(unlabel = None if evalb_lcfrs_prm.LABELED else 'X',
                excluded_labels = evalb_lcfrs_prm.DELETE_LABEL,
                equal_labels    = {l:ls[-1] for ls in eq_l for l in ls})
    excluded_words  = evalb_lcfrs_prm.DELETE_WORD
    equal_words     = {w:ws[-1] for ws in eq_w for w in ws}
    def filter_bottom(bt):
        bt_set = set()
        for bid, word, tag in g_bt:
            if equal_words:
                word = equal_words.get(word, word)
            if word in excluded_words:
                continue
            bt_set.add((bid, word, tag))
        return bt_set
    evalb = DiscoEvalb()
    with open(pred_fname) as f_pred, open(gold_fname) as f_gold:
        for p_line, g_line in zip(f_pred, f_gold):
            p_bt, p_td, p_rt = _read_dpenn(Tree.fromstring(p_line))
            g_bt, g_td, g_rt = _read_dpenn(Tree.fromstring(g_line))
            p_brackets = bracketing(p_bt, p_td, p_rt, False, **args)
            g_brackets = bracketing(g_bt, g_td, g_rt, False, **args)
            evalb.add(p_brackets, filter_bottom(p_bt), g_brackets, filter_bottom(g_bt))
    print(evalb)

class ExportWriter:
    def __init__(self):
        self._lines = []

    def add(self, bottom, top_down, root_id):
        n = len(self._lines) + 1
        self._lines.append(export_string(n, bottom, top_down, root_id))

    def dump(self, fname):
        with open(fname, 'w') as fw:
            fw.write('\n'.join(self._lines))

def export_string(sent_id, bottom, top_down, root_id):
    lines = f'#BOS {sent_id}\n'
    bottom_up = {}
    has_vroot = False
    node_dict = defaultdict(lambda: len(node_dict) + 500)
    for pid, td in top_down.items():
        if pid == root_id and td.label == 'VROOT':
            pid = 0
            has_vroot = True
        else:
            pid = node_dict[pid]
        for cid in td.children:
            bottom_up[cid] = pid
    for tid, word, tag in bottom:
        pid = bottom_up.pop(tid)
        lines += f'{word}\t\t\t{tag}\t--\t--\t{pid}\n'
    bottom_up = list(bottom_up.items())
    bottom_up.reverse()
    while bottom_up:
        cid, pid = bottom_up.pop()
        lines += f'#{node_dict[cid]}\t\t\t{top_down[cid].label}\t--\t--\t{pid}\n'
    if not has_vroot:
        lines+= f'#{node_dict[root_id]}\t\t\t{top_down[root_id].label}\t--\t--\t0\n'
    return lines + f'#EOS {sent_id}'