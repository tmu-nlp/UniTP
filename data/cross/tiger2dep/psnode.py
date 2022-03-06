#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# This file implements a node representation for nodes in a syntactic graph structure.
#
# author: Wolfgang Seeker
# 19/12/2012
#
    
class PSNode:
    def __init__(self,
                 sid = -1,
                 nid = -1,
                 form = None,
                 lemma = None,
                 pos = None,
                 morph = None,
                 head = -1,
                 label = None,
                 leaf = True,
                 origid = -1):
        self.sid = sid
        self.nid = nid
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.morph = morph
        self.head = head
        self.label = label
        self.leaf = leaf
        self.origid = origid
        
    def __str__(self):
        if self.morph:
            morphstr = '|'.join(f'{x}={y.lower()}' for x, y in self.morph)
        else:
            morphstr = '_'
        snid = (self.sid != -1 and f'{self.sid}_{self.nid}' or str(self.nid))
        return '\t'.join([snid, self.form, self.lemma, '_', self.pos, '_', morphstr, '_', str(self.head),'_',self.label,'_','_'])

    @property
    def tiger_sid(self):
        return f's{self.sid}'

    @property
    def dep_head(self):
        # head = f's{self.sid}_{self.head}' if self.head > 0 else None
        # return f's{self.sid}_{self.nid}', head
        return self.nid, self.head