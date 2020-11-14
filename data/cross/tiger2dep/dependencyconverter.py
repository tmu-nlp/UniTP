#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# This file implements a converter that converts TiGerXML trees to dependencies.
# Note that the TiGer conversion only works with version 2.1 of the TiGer corpus, release August 2007.
#
# author: Wolfgang Seeker
# 19/12/2012
#
# 30/10/2019: edit, adding 'interviews' to the possible datasets 

import itertools
import codecs
import sys
if __name__ == '__main__':
    from psnode import PSNode
else:
    from data.cross.tiger2dep.psnode import PSNode

class NoHeadFoundError(Exception):
    def __init__( self, sid, nid, pos ):
        self.sid = sid
        self.nid = nid
        self.pos = pos
        
    def __str__( self ):
        return repr('NoHeadFoundError occurred: sentence id: %d token id: %d poscat: %s' % (self.sid,self.nid,self.pos))

class MultipleHeadsFoundError(Exception):
    def __init__( self, sid, nid, pos ):
        self.sid = sid
        self.nid = nid
        self.pos = pos
        
    def __str__( self ):
        return repr('MultipleHeadsFoundError occurred: sentence id: %d token id: %d poscat: %s' % (self.sid,self.nid,self.pos))

        

class DependencyConverter:
    def __init__( self, manualhead_file, datasource, punctuation='deepest-common-ancestor', coordination='chain', ellipsis='resolve'):
        self.__options = {}
        self.__options['punctuation'] = punctuation
        self.__options['coordination'] = coordination
        self.__options['ellipsis'] = ellipsis
        self.__options['datasource'] = datasource
        
        self.manual_heads = {}
        self.__read_manual_heads(manualhead_file)
        self.coordphrases = ['CAC','CAP','CAVP','CCP','CNP','CO','CPP','CS','CVP','CVZ']
        self.headrules = { 'S':  [('s','HD',[])],
                           'VP': [('s','HD',[])],
                           'VZ': [('s','HD',[])],
                           'AVP':[('s','HD',[]),('s','PH',[]),('r','AVC',['ADV']),('l','AVC',['FM'])],
                           'AP': [('s','HD',[]),('s','PH',[])],
                           'DL': [('s','DH',[])],
                           'AA': [('s','HD',[])],
                           'ISU':[('l','UC',[])],
                           'PN': [('r','PNC',['NE','NN','FM','TRUNC','APPR','APPRART','CARD','VVFIN','VAFIN','ADJA','ADJD','XY','<EMPTY>'])],
                           'NM': [('r','NMC',['NN','CARD'])],
                           'MTA':[('r','ADC',['ADJA'])],
                           'PP': [('r','HD',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','AC',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','PH',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('l','NK',['PROAV'])],
                           'CH': [('l','UC',['FM','NE','XY','CARD'])],
                           'NP': [('l','HD',['NN']),('l','NK',['NN']),('r','HD',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT','<EMPTY>']),('r','NK',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT','<EMPTY>']),('r','PH',['NN','NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT','<EMPTY>']),('s','NK',[])],                       
                        }
        if self.__options['datasource'] == 'europarl':
            self.headrules['TOP'] = [('l','--',['VVFIN','VAFIN','VMFIN','VVIMP'])]

        if self.__options['datasource'] == 'smultron':
            self.headrules = { 'S': [('s','HD',[])],
                              'VP': [('s','HD',[])],
                              'VZ': [('s','HD',[])],
                              'AVP':[('s','HD',[]),('s','PH',[]),('r','AVC',['ADV']),('l','AVC',['FM'])],
                              'AP': [('s','HD',[]),('s','PH',[])],
                              'DL': [('s','DH',[])],
                              'AA': [('s','HD',[])],
                              'ISU':[('l','UC',[])],
                              'PN': [('r','PNC',['NE','NN','FM','TRUNC','APPR','APPRART','CARD','VVFIN','VAFIN','ADJA','ADJD','XY'])],
                              'MPN': [('r','PNC',['NE','NN','FM','TRUNC','APPR','APPRART','CARD','VVFIN','VAFIN','ADJA','ADJD','XY'])],
                              'NM': [('r','NMC',['NN','CARD'])],
                              'MTA':[('r','ADC',['ADJA'])],
                              'PP': [('r','HD',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','AC',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','PH',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('l','NK',['PROAV'])],
                              'CH': [('s','HD',[]),('l','UC',['FM','NE','XY','CARD','ITJ'])],
                              'NP': [('l','HD',['NN']),('l','NK',['NN']),('r','HD',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT']),('r','NK',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT']),('r','PH',['NN','NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT'])],              
                              'CAC':[('l','CJ',[])],
                              'CAP':[('l','CJ',[])],
                             'CAVP':[('l','CJ',[])],
                              'CCP':[('l','CJ',[])],
                              'CNP':[('l','CJ',[])],
                              'CO': [('l','CJ',[])],
                              'CPP':[('l','CJ',[])],
                              'CS': [('l','CJ',[])],
                              'CVP':[('l','CJ',[])],
                              'CVZ':[('l','CJ',[])]
                            }

        self.ellipsis_resolution = ['VVFIN','VAFIN','VMFIN','NN','NE','APPRART','APPR','APPO','PTKNEG','VVPP','VAPP','ADJD','PROAV','PPER','VVINF','VAINF','VMINF','ADJA','CARD','KOUS','PWAV','PTKANT','PIS','PDS','PWS','ADV','TRUNC','FM','ITJ']
        self.sentence_phrases = [ 'S' ]

        if self.__options['datasource'] == 'interviews':
            self.ellipsis_resolution.append('PIAT')
            
            self.coordphrases.append('S+')
            self.sentence_phrases.append('S-')

            #self.headrules['S-'] = [('s','HD',[]), ('s','SB',['NN',"PPER", "CARD", "PIS"]), ('s','OA',['NE' ])]
            self.headrules['NP'] = self.headrules['NP'][:-1]
    
    def convert( self, sentence ):
        """Convert a sentence"""
        if self.__options['punctuation'] == 'deepest-common-ancestor':
            self.__attach_punct_2(sentence)
        try:
            roots = [ node.nid for node in sentence if node.head == -1 and not node.leaf ]
            for rootid in roots:
                self.__convert(sentence,rootid)   
        except NoHeadFoundError as e:
            print(e, file = sys.stderr)
        except MultipleHeadsFoundError as e:
            print(e, file = sys.stderr)
        if self.__options['ellipsis'] == 'resolve':
            self.__resolve_ellipses(sentence)
        if self.__options['punctuation'] == 'easy':
            self.__attach_punct(sentence)
        self.__remove_phrases(sentence)
        if self.__options['ellipsis'] == 'keep':
            self.__move_ellipses(sentence)


    def convert_treebank( self, treebank ):
        """Convert a list of sentences"""
        for sentence in treebank:
            self.convert(sentence)


    def __read_manual_heads( self, filename ):
        """Read in manual head selection rules"""
        with open(filename) as fr:
            for line in fr:
                line = line.partition('%')[0].strip() # to get rid of prolog comments
                if line.startswith('manual_head(') and line.endswith(').'):
                    ids = [int(x) for x in line.partition('(')[2][:-2].split(',')]
                    if ids[0] not in self.manual_heads:
                        self.manual_heads[ids[0]] = {}            
                    self.manual_heads[ids[0]][ids[1]] = ids[2]
                # this part should not happen if elliptic heads are meant to stay
                if self.__options['ellipsis'] == 'resolve' and line.startswith('manual_head_no_ell(') and line.endswith(').'):
                    ids = [int(x) for x in line.partition('(')[2][:-2].split(',')]
                    if ids[0] not in self.manual_heads:
                        self.manual_heads[ids[0]] = {}            
                    self.manual_heads[ids[0]][ids[1]] = ids[2]
    

    def __manual_head( self, sentence, rootid, daughters ):
        """Return manually selected head for given phrase"""
        sid = sentence[0].sid
        origid = sentence[rootid-1].origid
        if sid in self.manual_heads and origid in self.manual_heads[sid]:
            headid = self.manual_heads[sid][origid]
            candidates = [ dau for dau in daughters if dau.nid == headid ]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise MultipleHeadsFoundError(sid,origid,sentence[rootid-1].pos)
            else:
                raise NoHeadFoundError(sid,origid,sentence[rootid-1].pos)
        return None


    def __match( self, daughters, label, poslist ):
        """Match a list of daughters against a heuristic to generate head candidates"""
        for pos in poslist:
            yield [ dau for dau in daughters if dau.label == label and dau.pos == pos ]
        if not poslist:
            yield [ dau for dau in daughters if dau.label == label ]
        

    def __find_head( self, sentence, rootid, daughters, pcat ):
        """Return head for given phrase"""
        # if there is a manually selected head, return it
        manhead = self.__manual_head(sentence,rootid,daughters)
        if manhead: return manhead
                
        # otherwise use heuristics to find the head of the phrase
        for dir,label,poslist in self.headrules.get(pcat,[]):
            for headcandidates in self.__match(daughters,label,poslist):
                if dir == 's' and len(headcandidates) == 1:
                    return headcandidates[0]
                elif dir == 'l' and headcandidates:
                    return headcandidates[0]
                elif dir == 'r' and headcandidates:
                    return headcandidates[-1]
                elif dir == 's' and len(headcandidates) > 1:
                    raise MultipleHeadsFoundError(sentence[0].sid,sentence[rootid-1].origid,sentence[rootid-1].pos)
           
        # still didn't find anything
        return None
                    

    def __introduce_NP( self, sentence, rootid ):
        """Reattach some daughters of a PP to the NP head"""
        if self.__options['datasource'] == 'smultron': return # smultron already annotates structured PPs
        daughters = [ d for d in sentence if d.head == rootid ]
        nphead = self.__find_head(sentence,rootid,daughters,'NP')
        if nphead:
            for d in [ d for d in daughters if d.label not in ['AC','PH','MO','CM'] and d.nid != nphead.nid ]:
                sentence[d.nid-1].head = nphead.nid
        

    def __convert_coordination_as_chain( self, sentence, rootid ):
        """Convert coordination as chain"""
        daughters = sorted([ d for d in sentence if d.head == rootid ], key= lambda d: d.nid )
        conjuncts, modifiers = self.__partition_list(lambda n: n.label in ['CJ','CD'],daughters)        
        leftdeps = []
        rightdeps = []
        newhead = None
        for cj in conjuncts:
            parents = [ t.pos for t in sentence if hasattr(t, "phraseHead") and t.phraseHead == cj.nid-1 ]
            if not newhead and cj.label == 'CJ' and cj.pos != 'TRUNC' and "S-" not in parents:
                newhead = cj
            elif not newhead:
                leftdeps.append(cj)
            else:
                rightdeps.append(cj)
        if not newhead:
            leftdeps = []
            rightdeps = []
            for cj in conjuncts:
                if not newhead and cj.label == 'CJ':
                    newhead = cj
                elif not newhead:
                    leftdeps.append(cj)
                else:
                    rightdeps.append(cj)            
        if newhead:
            if rightdeps:
                rightdeps[0].head = newhead.nid
                for i,cj in enumerate(rightdeps[1:]):
                    cj.head = rightdeps[i].nid
            if leftdeps:
                leftdeps = list(reversed(leftdeps))
                leftdeps[0].head = newhead.nid
                for i,cj in enumerate(leftdeps[1:]):
                    cj.head = leftdeps[i].nid
            if modifiers:
                for mod in modifiers:
                    mod.head = newhead.nid
        else:            
            raise NoHeadFoundError(sentence[0].sid,sentence[rootid-1].origid,sentence[rootid-1].pos)
        return newhead
            
            
    def __convert_coordination_as_bush( self, sentence, rootid ):
        daughters = sorted([ d for d in sentence if d.head == rootid ], key= lambda d: d.nid )
        conjuncts, modifiers = self.__partition_list(lambda n: n.label in ['CJ','CD'],daughters)        
        newhead = None
        for cj in conjuncts:
            if cj.label == 'CJ' and cj.pos != 'TRUNC':
               newhead = cj
               break
        if not newhead:
            for cj in conjuncts:
                if cj.label == 'CJ':
                    newhead = cj
                    break
        if newhead:
            for cj in conjuncts:
                if cj.nid != newhead.nid:
                    cj.head = newhead.nid            
            if modifiers:
                for mod in modifiers:
                    mod.head = newhead.nid
        else:
            raise NoHeadFoundError(sentence[0].sid,sentence[rootid-1].origid,sentence[rootid-1].pos)
        return newhead
            
        
    def __convert( self, sentence, rootid ):
        """Convert a given phrase"""
        # if current node is a terminal node, do nothing
        if sentence[rootid-1].leaf:
            return
        
        # convert all daughters of the current phrase before determining its own head
        for nodeid in [ node.nid for node in sentence if node.head == rootid and not node.leaf ]:
            self.__convert( sentence, nodeid )

        pcat = sentence[rootid-1].pos
        newhead = None

        if pcat in self.coordphrases: # treat coordination
            if self.__options['coordination'] == 'chain':
                newhead = self.__convert_coordination_as_chain(sentence,rootid)
            elif self.__options['coordination'] == 'bush':
                newhead = self.__convert_coordination_as_bush(sentence,rootid)
        else:
            # introduce NPs into PPs
            if pcat == 'PP':
                self.__introduce_NP(sentence,rootid)
            
            # find head for current phrase
            daughters = [ d for d in sentence if d.head == rootid ]    
            newhead = self.__find_head(sentence,rootid,daughters,pcat)    
            if not newhead:
                # introduce empty heads for verb and sentence phrases
                if pcat == 'VP' or pcat in self.sentence_phrases:
                    newhead = self.__make_empty_head(sentence,pcat)                    
                    sentence.append(newhead)
                    daughters = [ d for d in sentence if d.head == rootid ]    
                else: 
                    raise NoHeadFoundError(sentence[0].sid,sentence[rootid-1].origid,sentence[rootid-1].pos)
                    return
            
            # reattach siblings
            for d in daughters:
                if d.nid != newhead.nid:
                    sentence[d.nid-1].head = newhead.nid
                    
        # reattach head
        sentence[newhead.nid-1].head = sentence[rootid-1].head
        sentence[newhead.nid-1].label = sentence[rootid-1].label
        sentence[rootid-1].head = -1
        sentence[rootid-1].phraseHead = newhead.nid-1 # useful to find phrase types
    

    def __make_empty_head( self, sentence, pcat ):
        """Make new empty head node"""
        if pcat == 'S':
            morph = [('vtype','fin')]
        elif pcat == 'VP':
            morph = [('vtype','inf')]
        emptynode = PSNode(form = '<empty>',
                           lemma = '<empty>',
                           pos = '<EMPTY>',
                           morph = morph,
                           label = '_',
                           sid = sentence[0].sid,
                           nid = len(sentence) + 1)
        return emptynode
    
    
    def __partition_list( self, pred, iterable ):
        """Partition list into list that contains all elements that fullfil pred and rest"""
        t1, t2 = itertools.tee(iterable)
        t1 = [x for x in t1 if pred(x)]
        t2 = [x for x in t2 if not pred(x)]
        return t1, t2


    def __attach_punct( self, sentence ):
        """Attach punctuation to the preceeding word"""
        for node in sentence:
            if node.pos.startswith('$') and node.head == -1:
                if node.nid > 1:
                    node.head = node.nid-1
                else:
                    try:
                        node.head = next(itertools.dropwhile(lambda x: x.pos.startswith('$'), sentence)).nid
                    except StopIteration:
                        pass


    def __ancestors( self, sentence, node ):
        """Generator to return the ancestors of a node starting from the head"""
        current = node.nid
        while current not in [-1,0]:
            yield sentence[current-1].head
            current = sentence[current-1].head


    def __clausal_ancestors( self, sentence, node ):
        """Generator to return ancestors of a node starting with the head
        Only ancestors up to the next finite verb are return
        """
        for ancestor in self.__ancestors(sentence,node):
            #print >> sys.stderr, ancestor
            yield ancestor
            if sentence[ancestor-1].pos.endswith('FIN'):
                return
    

    def __deepest_common_ancestor( self, sentence, left, right ):
        """Returns the deepest common ancestor of two given nodes"""
        if left.nid == right.nid:
            return left.nid
        ancestors_left = list(self.__ancestors(sentence,left))
        for aid in self.__ancestors(sentence,right):
            if aid in ancestors_left:
                return aid
        

    def __next_word( self, tokens ):
        try:
            return next(itertools.dropwhile(lambda x: x.pos.startswith('$') or not x.leaf, tokens))
        except StopIteration:
            return None
        

    def __attach_punct_2( self, sentence ):
        """Attach punctuation more sensibly. Operates on phrase-structure tree"""
        roots = [ d for d in sentence if d.head == -1 and not d.leaf ]
        for node in sentence:
            # for enclosing punctuation try to find the corresponding symbol and attach them to the node that covers the span within
            if node.pos == '$(' and node.form in ['``','(','`']:
                complement = None
                if node.form == '``':
                    complement = [ n.nid for n in sentence[node.nid-1:] if n.form == "''" ]
                elif node.form == '(':
                    complement = [ n.nid for n in sentence[node.nid-1:] if n.form == ")" ]
                elif node.form == '`':
                    complement = [ n.nid for n in sentence[node.nid-1:] if n.form == "'" ]
                if complement:
                    right = self.__next_word(sentence[node.nid:complement[0]-1])
                    left = self.__next_word(reversed(sentence[node.nid:complement[0]-1]))
                    if left and right:
                        node.head = self.__deepest_common_ancestor(sentence,left,right)
                        sentence[complement[0]-1].head = node.head
                        continue
            # default case
            if node.pos.startswith('$') and node.leaf and node.head == -1: # attach it to the deepest common ancestor of the left and the right neighbor
                right = self.__next_word(sentence[node.nid:])
                left = self.__next_word(reversed(sentence[:node.nid-1]))
                if not left and not right: # don't do anything to one word sentences (this should never happen)
                    pass
                elif not left: # attach punct to first root if it is the first token
                    try:
                        node.head = roots[0].nid
                    except IndexError:
                        node.head = -1
                elif not right: # attach punct to last root if it is the last token
                    try:
                        node.head = roots[-1].nid
                    except IndexError:
                        node.head = -1                   
                else: 
                    node.head = self.__deepest_common_ancestor(sentence,left,right)
                


    def __remove_phrases( self, sentence ):
        """Remove phrasal nodes from the sentence structure"""
        # delete phrasal nodes
        for i in reversed(range(len(sentence))):
            if not sentence[i].leaf:
                del sentence[i]
        # renumber empty nodes
        newnumbers = {}
        for i,node in enumerate(sentence):
            if self.__is_empty_head(node):
                oldnid = node.nid
                node.nid = sentence[i-1].nid+1
                newnumbers[oldnid] = node.nid
        if newnumbers:
            for node in sentence:
                if node.head in newnumbers:
                    node.head = newnumbers[node.head]            
        # renumber heads
        for node in sentence:
            if node.head == -1:
                node.head = 0

    
    def __is_empty_head( self, node ):
        """Tell whether given node is empty head"""
        return node.form == '<empty>' and node.lemma == '<empty>' and node.pos == '<EMPTY>'
        

    def __resolve_ellipses( self, sentence ):
        """Resolve elliptic phrases by making one of the dependents the head"""
        roots = [ node.nid for node in sentence if node.head == -1 ]
        for rootid in roots:
            self.__find_ellipsis_head(sentence,rootid)
        for i in reversed(range(len(sentence))):
            if self.__is_empty_head(sentence[i]):
                del sentence[i]


    def __find_ellipsis_head( self, sentence, rootid ):
        """Find the head for the current elliptical construction"""
        daughterids = [ node.nid for node in sentence if node.head == rootid ]    
        # leaf found
        if not daughterids:
            return
        # inner node found
        for nodeid in daughterids:
            self.__find_ellipsis_head( sentence, nodeid )
        
        if self.__is_empty_head(sentence[rootid-1]):     
            daughters = [ d for d in sentence if d.head == rootid ]
            newhead = None
            for pos in self.ellipsis_resolution:
                headcandidates = [ d for d in daughters if d.pos == pos ]
                if headcandidates:
                    newhead = headcandidates[0]
                    break
            
            for d in daughters:
                sentence[d.nid-1].head = newhead.nid

            # reattach head
            sentence[newhead.nid-1].head = sentence[rootid-1].head
            sentence[newhead.nid-1].label = sentence[rootid-1].label
            sentence[rootid-1].head = -1


    def __move_ellipses( self, sentence ):
        """S-ellipsis are moved to second position in clause, VP-ellipsis to last position
        a clause is defined as the set of all daughters excluding finite verbs and other ellipsis
        """
        # go through ellipses deepest first
        for _,ell in sorted([ (self.__depth(sentence,node),node) for node in sentence if self.__is_empty_head(node) ],reverse=True):
            daughters = [ d for d in sentence if d.head == ell.nid and not d.pos.endswith('FIN') and not d.pos.startswith('$') ]
            origid = ell.nid
            newnid = -1    
            if ('vtype','inf') in ell.morph or ell.label == 'RC' or len([ d for d in sentence if d.head == ell.nid and d.label in ['CP','CM'] ]) > 0 or (ell.head > 0 and sentence[ell.head-1].label == 'RC' ): # move to last position
                last = daughters[-1]
                lastspan = [ node for node in sentence if last.nid in list(self.__clausal_ancestors(sentence,node)) or node.nid == last.nid ]
                newnid = lastspan[-1].nid+1
            else: # move to second position
                if len(daughters) == 0: # this occurs with 11330 and 36241, the second of which seems to be incorrectly annotated (should be unlike constituent coordination)
                    # print >> sys.stderr, sentence[0].sid 
                    continue
                first = daughters[0]
                firstspan = [ node for node in sentence if first.nid in list(self.__clausal_ancestors(sentence,node)) or node.nid == first.nid ]
                newnid = firstspan[-1].nid+1
            for node in sentence:                    
                if node.nid >= newnid and node.nid < origid:
                    node.nid += 1
                if node.head >= newnid and node.head < origid:
                    node.head += 1
                elif node.head == origid:
                    node.head = newnid
            ell.nid = newnid
            sentence.sort(key=lambda x: x.nid)


    def __depth( self, sentence, node ):
        return len([ aid for aid in self.__ancestors(sentence,node) ])

        
