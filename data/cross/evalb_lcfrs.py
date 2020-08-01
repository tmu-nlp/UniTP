from collections import Counter
from utils.math_ops import bit_fanout

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

class DiscoEvalb:
    def __init__(self):
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
        if p_tags is None:
            if p_brackets is None:
                self._sent_lines.append(f'{len(self._sent_lines) + 1:5d}')
        else:
            tag_match = len(p_tags & g_tags)
            p_tag_count = len(p_tags)
            g_tag_count = len(g_tags)
            self._total_matched_pos += tag_match
            self._total_pos += g_tag_count
            if p_brackets is None:
                self._sent_lines.append(f'{len(self._sent_lines) + 1:5d}' + 45 * ' ' + '%3d  %3d' % (p_tag_count, tag_match))
        if p_brackets is None:
            self._missing += 1
            return

        bracket_match = sum((g_brackets & p_brackets).values())
        p_num_brackets = sum(p_brackets.values())
        g_num_brackets = sum(g_brackets.values())
        g_disc_brackets = Counter({(lb, fo):ct for (lb, fo), ct in g_brackets.items() if bit_fanout(fo) > 1})
        p_disc_brackets = Counter({(lb, fo):ct for (lb, fo), ct in p_brackets.items() if bit_fanout(fo) > 1})
        disc_bracket_match = sum((g_disc_brackets & p_disc_brackets).values())
        g_disc_num_brackets = sum(g_disc_brackets.values())
        p_disc_num_brackets = sum(p_disc_brackets.values())
        if g_brackets == p_brackets:
            self._total_exact += 1
        if g_disc_num_brackets:
            self._disc_sents += 1
            if g_disc_brackets == p_disc_brackets:
                self._disc_exact += 1
        sent_prec, sent_rec, sent_fb1 = _scores(bracket_match, p_num_brackets, g_num_brackets)
        self._sent_lines.append(
            "%5d  %6.2f  %6.2f  %6.2f    %3d    %3d  %3d  %3d  %3d" % (
                len(self._sent_lines) + 1, sent_prec, sent_rec, sent_fb1,
                bracket_match, g_num_brackets, p_num_brackets, p_tag_count, tag_match))
        self._total_match += bracket_match
        self._total_pred += g_num_brackets
        self._total_gold += p_num_brackets
        self._disc_match += disc_bracket_match
        self._disc_pred += p_disc_num_brackets
        self._disc_gold += g_disc_num_brackets

    def __str__(self):
        tp, tr, tf, dp, dr, df = self.summary()
        line = ' sent.  prec.   rec.    F1     match   gold test tags match\n'
        line += '===========================================================\n'
        line += '\n'.join(self._sent_lines)
        line += '\n===========================================================\n\n\n'
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