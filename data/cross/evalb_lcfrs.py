# Authors: Wolfgang Maier <maierw@hhu.de>,
# Andreas van Cranenburgh <a.w.vancranenburgh@uva.nl>
# Version: January 24, 2014
# Modified By zchen0420@github, 2020

from collections import Counter, defaultdict
from utils.math_ops import bit_fanout
from utils.param_ops import HParams

def make_eq_fn(eq):
    def fn(x):
        return eq.get(x, x)
    return fn

def read_param(filename):
    validkeysonce = ('DEBUG', 'MAX_ERROR', 'CUTOFF_LEN', 'LABELED',
                     'DISC_ONLY', 'TED', 'DEP')
    param = {'DEBUG': 0, 'MAX_ERROR': 10, 'CUTOFF_LEN': 9999,
             'LABELED': 1, 'DELETE_LABEL_FOR_LENGTH': set(),
             'DELETE_LABEL': set(), 'DELETE_WORD': set(),
             'EQ_LABEL': {}, 'EQ_WORD': {},
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
                param[key][b] = c
            else:
                raise ValueError('unrecognized parameter key: %s' % key)

    if param['LABELED']:
        label_fn = make_eq_fn(param['EQ_LABEL'])
    else:
        label_fn = lambda _: 'X'
    param['word_fn'] = make_eq_fn(param['EQ_WORD'])
    param['label_fn'] = label_fn
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

from data.cross.dptb import read_tree, Tree
def conti_matches(p_tree, g_tree):
    g_bt, g_td = read_tree(g_tree, adjust_fn = None)
    p_bt, p_td = read_tree(p_tree, adjust_fn = None)
    bracket_match, _, p_num_brackets, _, _, g_num_brackets, _, _ = disco_matches(bracketing(p_bt, p_td), bracketing(g_bt, g_td))
    return bracket_match, p_num_brackets, g_num_brackets

def ndd(brackets):
    disc_brackets = {}
    num_brackets = num_disc_brackets = 0
    for key, ct in brackets.items():
        _, fo = key
        num_brackets += ct
        if bit_fanout(fo) > 1:
            num_disc_brackets += ct
            disc_brackets[key] = ct
    return num_brackets, num_disc_brackets, Counter(disc_brackets)

def mul_fan_ndd(brackets, multibs):
    mul_num = defaultdict(int)
    mul_uni = defaultdict(dict)
    fan_num = defaultdict(int)
    fan_uni = defaultdict(dict)
    for key, ct in brackets.items():
        f = bit_fanout(key[1])
        m = multibs[key]
        mul_num[m] += ct
        mul_uni[m][key] = ct
        fan_num[f] += ct
        fan_uni[f][key] = ct
    return mul_num, mul_uni, fan_num, fan_uni

def disco_matches(p_brackets, g_brackets):
    g_num_brackets, g_disc_num_brackets, g_disc_brackets = ndd(g_brackets)
    p_num_brackets, p_disc_num_brackets, p_disc_brackets = ndd(p_brackets)
    bracket_match = sum((g_brackets & p_brackets).values())
    disc_bracket_match = sum((g_disc_brackets & p_disc_brackets).values())
    return bracket_match, disc_bracket_match,\
        p_num_brackets, p_disc_num_brackets, p_disc_brackets,\
        g_num_brackets, g_disc_num_brackets, g_disc_brackets

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
        self._mul_match = defaultdict(int)
        self._mul_gold = Counter()
        self._mul_pred = Counter()
        self._fan_match = defaultdict(int)
        self._fan_gold = Counter()
        self._fan_pred = Counter()

    def add(self, p_brackets, p_multib, p_tags, g_brackets, g_multib, g_tags):
        self._total_sents += 1
        g_tag_count = len(g_tags)

        g_num_brackets, g_disc_num_brackets, g_disc_brackets = ndd(g_brackets)
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

        if g_brackets == p_brackets:
            self._total_exact += 1

        p_num_brackets, p_disc_num_brackets, p_disc_brackets = ndd(p_brackets)
        bracket_match = sum((g_brackets & p_brackets).values())
        disc_bracket_match = sum((g_disc_brackets & p_disc_brackets).values())
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

        p_mdn, p_mdb, p_fon, p_fob = mul_fan_ndd(p_brackets, p_multib)
        g_mdn, g_mdb, g_fon, g_fob = mul_fan_ndd(g_brackets, g_multib)
        self._mul_pred += p_mdn
        self._mul_gold += g_mdn
        self._fan_pred += p_fon
        self._fan_gold += g_fon
        for m in p_mdb.keys() & g_mdb.keys():
            self._mul_match[m] += sum((Counter(p_mdb[m]) & Counter(g_mdb[m])).values())
        for f in p_fob.keys() & g_fob.keys():
            self._fan_match[f] += sum((Counter(p_fob[f]) & Counter(g_fob[f])).values())

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
        line += '    Disc. EX  : %6.2f %%\n\n' % (100 * self._disc_exact / self._disc_sents if self._disc_sents else 0)
        line += 'Multi-branching LP LR F1 (P/G/M)\n'
        for m, (p, r, f, mm, mp, mg) in self.summary_multib():
            line += f'    {m:3d}  |  {p:6.2f}%  {r:6.2f}%  {f:6.2f}%  ({mp}/{mg}/{mm})\n'
        line += '\nFan-out LP LR F1 (P/G/M)\n'
        for fo, (p, r, f, fm, fp, fg) in self.summary_fanout():
            line += f'    {fo:3d}  |  {p:6.2f}%  {r:6.2f}%  {f:6.2f}%  ({fp}/{fg}/{fm})\n'
        return line

    def summary(self):
        total = _scores(self._total_match, self._total_pred, self._total_gold)
        disc  = _scores( self._disc_match,  self._disc_pred,  self._disc_gold)
        return total + disc

    def summary_multib(self):
        for m in sorted(self._mul_pred.keys() | self._mul_gold.keys()):
            mm = self._mul_match[m]
            mp = self._mul_pred[m]
            mg = self._mul_gold[m]
            yield m, _scores(mm, mp, mg) + (mm, mp, mg)
    
    def summary_fanout(self):
        for fo in sorted(self._fan_pred.keys() | self._fan_gold.keys()):
            fm = self._fan_match[fo]
            fp = self._fan_pred[fo]
            fg = self._fan_gold[fo]
            yield fo, _scores(fm, fp, fg) + (fm, fp, fg)
            
    @property
    def total_missing(self):
        return self._total_sents, self._missing

def continuous_evalb(pred_fname, gold_fname, prm_fname):
    evalb_lcfrs_prm = read_param(prm_fname)
    evalb = DiscoEvalb()
    with open(pred_fname) as f_pred, open(gold_fname) as f_gold:
        for p_line, g_line in zip(f_pred, f_gold):
            try:
                p_bt, p_td = read_tree(Tree.fromstring(p_line))
                g_bt, g_td = read_tree(Tree.fromstring(g_line))
                p_bt, p_td = new_word_label(p_bt, p_td, word_fn = evalb_lcfrs_prm.word_fn, label_fn = evalb_lcfrs_prm.label_fn)
                g_bt, g_td = new_word_label(g_bt, g_td, word_fn = evalb_lcfrs_prm.word_fn, label_fn = evalb_lcfrs_prm.label_fn)
                filter_words(p_bt, p_td, evalb_lcfrs_prm.DELETE_WORD)
                filter_words(g_bt, g_td, evalb_lcfrs_prm.DELETE_WORD)
                p_brackets, p_multibs = bracketing(p_bt, p_td, False, evalb_lcfrs_prm.DELETE_LABEL)
                g_brackets, g_multibs = bracketing(g_bt, g_td, False, evalb_lcfrs_prm.DELETE_LABEL)
                evalb.add(p_brackets, p_multibs, set(p_bt), g_brackets, g_multibs, set(g_bt))
            except:
                continue
    return evalb

def eval_disc(p_lines, g_lines, evalb_lcfrs_prm):
    p_bt, p_td = parse_export_sample(p_lines, C_VROOT)
    g_bt, g_td = parse_export_sample(g_lines, C_VROOT)
    p_bt, p_td = new_word_label(p_bt, p_td, word_fn = evalb_lcfrs_prm.word_fn, label_fn = evalb_lcfrs_prm.label_fn)
    g_bt, g_td = new_word_label(g_bt, g_td, word_fn = evalb_lcfrs_prm.word_fn, label_fn = evalb_lcfrs_prm.label_fn)
    filter_words(p_bt, p_td, evalb_lcfrs_prm.DELETE_WORD)
    filter_words(g_bt, g_td, evalb_lcfrs_prm.DELETE_WORD)
    p_brackets, p_multibs = bracketing(p_bt, p_td, False, evalb_lcfrs_prm.DELETE_LABEL)
    g_brackets, g_multibs = bracketing(g_bt, g_td, False, evalb_lcfrs_prm.DELETE_LABEL)
    return p_brackets, p_multibs, set(p_bt), g_brackets, g_multibs, set(g_bt)

def discontinuous_evalb(pref_fname, gold_fname, prm_fname):
    evalb_lcfrs_prm = read_param(prm_fname)
    evalb = DiscoEvalb()
    for p_lines, g_lines in zip(read_export_samples(pref_fname), read_export_samples(gold_fname)):
        evalb.add(*eval_disc(p_lines, g_lines, evalb_lcfrs_prm))
    return evalb

class ExportWriter:
    def __init__(self):
        self._lines = []

    def add(self, bottom, top_down, root_id = 0):
        n = len(self._lines) + 1
        self._lines.append(export_string(n, bottom, top_down, root_id))

    def dump(self, fname):
        with open(fname, 'w') as fw:
            fw.write('\n'.join(self._lines))

from data.cross import TopDown, C_VROOT, bracketing, filter_words, new_word_label
def read_export_samples(fname):
    sample = None
    with open(fname) as fr:
        for line in fr:
            if line.startswith('#BOS'):
                sample = [line]
            elif line.startswith('#EOS'):
                sample.append(line)
                yield sample
                sample = None
            else:
                sample.append(line)

from data.cross.tiger import get_inode, C_TIGER_NT_START
def tiger_inode(node):
    if node[0] == '#':
        core = node[1:]
        if core.isdigit():
            return get_inode(int(core))

def parse_export_sample(lines, fallback = None, dptb_split = False):
    bottom = []
    top_down = defaultdict(list)
    non_terminals = {}
    for line in lines[1:-1]:
        if dptb_split:
            cid_or_word, label, _, ftag, pid = line.split()
        else:
            cid_or_word, _, _, label, _, _, pid = line.split('\t')
        pid = get_inode(int(pid))
        if (cid := tiger_inode(cid_or_word)) and cid < 0:
            non_terminals[cid] = label
            top_down[pid].append(cid)
        else:
            assert cid is None or cid != 0
            nid = len(bottom) + 1
            bottom.append((nid, cid_or_word, label))
            top_down[pid].append(nid)
    if len(top_down[0]) == 1 and fallback is None: # for DPTB's
        assert non_terminals
        root = top_down.pop(0).pop() # == 1
        top_down[0] = top_down.pop(root)
        non_terminals[0] = non_terminals.pop(root)
    elif isinstance(fallback, str): # 0 is innate for .export, add VROOT
        non_terminals[0] = fallback
    for pid, label in non_terminals.items():
        top_down[pid] = TopDown(label, {n: None for n in top_down[pid]})
    return bottom, top_down

def export_string(sent_id, bottom, top_down, root_id = 0):
    lines = f'#BOS {sent_id}\n'
    bottom_up = {}
    has_vroot = False
    assert len(bottom) < C_TIGER_NT_START, 'TODO: let discodop support large bottom'
    assert len(top_down) < C_TIGER_NT_START, 'TODO: let discodop support large top_down'
    node_dict = defaultdict(lambda: len(node_dict) + C_TIGER_NT_START)
    for pid, td in top_down.items():
        if pid == root_id and td.label == C_VROOT:
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

class ExportReader:
    def __init__(self, fname, *la, **ka):
        from tqdm import tqdm
        samples = []
        indices = []
        for lines in tqdm(read_export_samples(fname)):
            _, head = lines[0].split()
            _, tail = lines[-1].split()
            assert head == tail
            head = int(head)
            samples.append(parse_export_sample(lines, *la, **ka))
            indices.append(head)
        self._samples = samples
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    @property
    def indices_is_continuous(self):
        return all(i+1==j for i, j in enumerate(self._indices))

    def __getitem__(self, idx):
        return self._samples[idx]