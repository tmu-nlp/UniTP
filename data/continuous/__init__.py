from data import XRB2brackets, no_backslash, RB2brackets, SUB
from nltk.tree import Tree
from data.cross import draw_str_lines as __draw
from data.cross.dptb import direct_read as __read
from random import random

draw_str_lines = lambda tree, **kwargs: __draw(*__read(tree), **kwargs)

def ptb_update_word(tag_unary, word, _):
    if word != (mord := no_backslash(word)):
        tag_unary[0] = mord

def ctb_update_tag(tag_unary, _, tag):
    # CTB has suffixes e.g., -2 -SHORT, KTB has others
    if (jo := tag.find('-')) > 0:
        tag_unary.set_label(tag[:jo])

def ktb_update_tag(tag_unary, word, tag):
    # KTB/NPCMJ has i.e., ;{extra information}
    if tag == 'NUMCLP' and word == 'NUM':
        tag_unary[0] = Tree(word, ['31'])
        return # sorry, I take this advantage
        
    mag = tag[:jo] if (jo := tag.find(';')) > 0 else tag
            
    if mag in ('ADJP', 'ADVP', 'PP', 'NP', 'LST'):
        mag = mag[:-1]
    else: # tag = {, 'PRN', 'COMMENT', 'FS', 'LST'}
        mag = {'FS': 'INTJ', 'PRN': 'NPR'}.get(mag, mag)
    
    if tag != mag:
        tag_unary.set_label(mag)

def add_subs(tree, prob):
    if isinstance(tree, Tree) and syn_tree(tree):
        any_modified, children = False, []
        for subtree in tree:
            p_modified = add_subs(subtree, prob)
            children.append(p_modified)
            if p_modified is not subtree:
                any_modified = True
        if any_modified:
            tree = Tree(tree.label(), children)

        if ((nc := len(tree)) > 1) and prob:
            subs = [i for i in range(1, nc) if random() < prob]
            if 0 < len(subs) < nc - 1:
                label = tree.label()
                sub_l = label if label[0] == SUB else (SUB + label)
                children = []
                for start, end in zip([0] + subs, subs + [nc]):
                    if start + 1 == end:
                        children.append(tree[start])
                    else:
                        children.append(Tree(sub_l, tree[start:end]))
                tree = Tree(label, children)
    return tree

def add_efficient_subs(tree, max_range = None):
    original, modified = 1, 0
    if len(tree) > 1:
        label, children = tree.label(), []
        for t in tree:
            (o, m), t = add_efficient_subs(t, max_range)
            original += o;  modified += m
            children.append(t)
        heights = set(t.height() for t in children)
        if max_range is None:
            upper = max(heights)
        else:
            upper = min(heights) + max_range
            upper = min(upper, max(heights))
        for ht in range(min(heights), upper):
            new_children = []
            for child in children:
                if not new_children or child.height() > ht:
                    new_children.append([child])
                else:
                    if new_children[-1][-1].height() > ht:
                        new_children.append([child])
                    else:
                        new_children[-1].append(child)
            children = []
            for group in new_children:
                if len(group) == 1:
                    children.append(group.pop())
                else:
                    modified += 1
                    children.append(Tree(SUB + label, group))
            tree = Tree(label, children)
    return (original, modified), tree

syn_tree = lambda t: t.height() > 2

def syn_label(label):
    for eid, char in enumerate(label):
        cid = ord(char)
        if cid < 65 or cid > 90:
            return label[:eid]
    return label

def bottom_paths(lexical, tree, length):
    paths = []
    for i in range(length): # diff from neural index
        path = tree.leaf_treeposition(i)[:-1]
        if lexical:
            if len(tree[path[:-1]]) == 1:
                path = path[:-1], False
            else:
                path = path, True
        else:
            path = path, False
        paths.append(path)
    return tuple(paths)

def adjust_label(mtree, word_trace):
    for subtree in mtree.subtrees(syn_tree):
        label = subtree.label()
        if label in ('ADVP|PRT', 'PRT|ADVP'):
            mabel = 'ADVP+PRT'
        elif word_trace and label == 'multi-sentence':
            continue
        else:
            mabel = syn_label(label)

        if label != mabel:
            subtree.set_label(mabel)

def remove_repeated_unary(tree):
    if isinstance(tree, Tree) and syn_tree(tree):
        while len(tree) == 1 and (subtree := tree[0]).label() in (tree.label(), '') and syn_tree(subtree):
            tree.pop()
            tree.extend(subtree)
            tree = subtree
        for subtree in tree:
            remove_repeated_unary(subtree)

def remove_trace(mtree, update_word_tag_fn, word_trace):
    for i, word in sorted(enumerate(mtree.leaves()), reverse = True):
        tag_path = mtree.leaf_treeposition(i)[:-1] # leaf must be unary
        if (tag := mtree[tag_path].label()) == '-NONE-':
            remove_tag = True
        elif word_trace:
            remove_tag = (word[0] == '*' and (word[-1] == '*' or '*' in word[1:] and word[-1].isdigit())) or tag == 'COMMENT'
        else:
            remove_tag = False
        if remove_tag:
            syn_path = tag_path[:-1]
            if len(mtree[syn_path]) == 1: # NP -NONE- *-1 then cascade
                while syn_path and len(mtree[syn_path[:-1]]) == 1:
                    syn_path = syn_path[:-1]
                del mtree[syn_path]
            else: # more than one child without cascade
                # NP (SBAR) (-NONE- *-1)
                del mtree[tag_path]
        elif callable(update_word_tag_fn):
            update_word_tag_fn(mtree[tag_path], word, tag)


from utils import do_nothing
class Signal:
    @classmethod
    def set_binary(cls):
        from data.continuous.binary import tree_loc_path_length, signals
        cls.b_prepare = tree_loc_path_length
        cls.b_signals = signals

    @classmethod
    def set_multib(cls):
        from data.continuous.multib import tree_path, signals
        cls.m_prepare = tree_path
        cls.m_signals = signals

    @classmethod
    def set_char(cls):
        from data.continuous.multib import char_signals
        cls.m_char_signals = char_signals

    def serialize(self, efficient = True):
        data = [self._tree, self._lexi]
        if efficient:
            ort, adt = self.original_and_additional
            data += [ort, adt]
            if adt:
                data.append(self.efficient_tree)
        return tuple(data)

    @classmethod
    def instantiate(cls, args):
        signal = cls(*args[:2])
        if args[2:]:
            etree = args[4] if args[3] else args[0]
            signal._esub = args[2:4], etree
        return signal

    @classmethod
    def from_ptb(cls, tree):
        # get rid of tree traces e.g. (-NONE- *)
        remove_trace(tree, ptb_update_word, False)
        adjust_label(tree, False)
        remove_repeated_unary(tree)
        tree.collapse_unary(collapseRoot = True)
        return cls(tree, True)

    @classmethod
    def from_ctb(cls, tree):
        remove_trace(tree, ctb_update_tag, False)
        adjust_label(tree, False)
        remove_repeated_unary(tree)
        tree.collapse_unary(collapseRoot = True)
        return cls(tree, True)

    @classmethod
    def from_ktb(cls, tree):
        # the are special word traces
        remove_trace(tree, ktb_update_tag, True)
        adjust_label(tree, True)
        remove_repeated_unary(tree)
        tree.collapse_unary(collapseRoot = True)
        return cls(tree, True)

    @classmethod
    def from_sstb(cls, tree):
        for sub in tree.subtrees(lambda x: x.height() == 2):
            if len(sub) > 1: # caused by b'\xc2\xa0'
                word = ''.join(sub)
                sub.clear()
                sub.append(word)
            else:
                word = sub[0]
            if '\/' in word:
                word = word.replace('\/', '/')
            if (mord := XRB2brackets(word)) != word:
                sub[0] = mord
        return cls(tree, False)

    def __init__(self, tree, lexical):
        bottom = tree.pos()
        self._tree = tree
        self._lexi = lexical
        self._blen = len(bottom)
        self._esub = None
        self._original  = [None, None]
        self._efficient = [None, None]
        self._charater  = [None, None]
        word, self._tag = zip(*bottom)
        self._word = tuple(RB2brackets(w) for w in word)

    @property
    def original_and_additional(self):
        if self._esub is None:
            self._esub = add_efficient_subs(self._tree)
        return self._esub[0]

    @property
    def efficient_tree(self):
        self.original_and_additional
        return self._esub[1]

    def char_segment(self, max_subword_height = None, more_sub = 0, joint = True):
        if all(len(w) == 1 for w in self._word):
            return [] if joint else [list(range(len(self._word) + 1))]
            
        if randomized := (more_sub and (max_subword_height is None or max_subword_height > 2)):
            tree, paths = Signal.m_char_signals(self._word, more_sub, max_subword_height)
        elif empty := (self._charater[1] is None):
            tree, paths = Signal.m_char_signals(self._word, max_subword_height = 2)
        if randomized or empty:
            _, bs = Signal.m_signals(tree, paths, joint = joint)
            bs = bs[:1]
            if not randomized:
                self._charater[1] = bs
        else:
            bs = self._charater[1]
        return bs

    def binary(self, factor = None, efficient = False, more_sub = 0, every_n = 1, *largs, **kwargs):
        return Signal.b_signals(factor, *self._prepare(True, efficient, more_sub), every_n, *largs, **kwargs)

    def multib(self, efficient = False, more_sub = 0, *largs, **kwargs):
        return Signal.m_signals(*self._prepare(False, efficient, more_sub), *largs, **kwargs)

    def _prepare(self, binary, efficient, more_sub):
        multib = not binary
        if more_sub:
            tree = self.efficient_tree if efficient else self._tree
            tree = add_subs(tree, more_sub)
            prepare = Signal.b_prepare if binary else Signal.m_prepare
            return prepare(self._lexi, tree, self._blen)
        elif efficient:
            if self._efficient[multib] is None:
                prepare = Signal.b_prepare if binary else Signal.m_prepare
                self._efficient[multib] = prepare(self._lexi, self.efficient_tree, self._blen)
        elif self._original[multib] is None:
            prepare = Signal.b_prepare if binary else Signal.m_prepare
            self._original[multib] = prepare(self._lexi, self._tree, self._blen)
        return (self._efficient if efficient else self._original)[multib]

    @property
    def tree(self):
        return self._tree

    @property
    def word(self):
        return self._word
    
    @property
    def tag(self):
        return self._tag

    @property
    def max_height(self):
        return self._blen

    def char_to_idx(self, c2i):
        idx = []
        for wd in self._word:
            idx.extend(c2i(w) for w in wd)
        return idx

    def word_to_idx(self, w2i):
        return [w2i(w) for w in self._word]
    
    def tag_to_idx(self, t2i):
        return [t2i(t) for t in self._tag]


def flatten_children(nodes):
    children = []
    for x in nodes:
        if isinstance(x, Tree):
            children.append(x)
        else:
            children.extend(x)
    return children