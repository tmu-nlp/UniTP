
def locations(node, bottom, top_down, consumed_top_down):
    locs = []
    children = consumed_top_down[node] if node in consumed_top_down else top_down[node].children
    for cid in children:
        if cid in bottom:
            locs.append(bottom.index(cid))
        else:
            locs.extend(locations(cid, bottom, top_down, consumed_top_down))
    return locs

from random import random
def _positional_right(cid, num_children, factor):
    position = 1 - (cid + 0.5) / num_children
    if position == factor == 0.5:
        right = random() > 0.5
    else:
        right = position > factor # T[TfF]F
    return right

def _new_dep(dependency, d_node, h_node, new_node):
    head = dependency.pop(h_node)
    head.children.update(dependency.pop(d_node).children)
    dependency[new_node] = head

from utils.math_ops import lr_gen
def binary_hash(bottom,
                top_down,
                completed_nodes,
                some_or_all,
                lean_joint,
                factor,
                sub_suffix,
                get_soft_right,
                dependency):
    bottom_up = {}
    right_hash = {}
    swappable_locations = []
    joint_rset = []
    bottom_flag = [True for _ in bottom]
    soft_right = {}
    consumed_top_down = {}
    depend_on = lambda d_node, h_node: dependency[d_node].label in dependency[h_node].children
    for p_node, td in top_down.items():
        p_node_complete = p_node in completed_nodes
        not_enough_child = not some_or_all(node in bottom for node in td.children)
        if p_node_complete or not_enough_child:
            if get_soft_right and not p_node_complete and not_enough_child:
                existing_nodes = []
                shadow_locations = []
                for node in td.children:
                    # import pdb; pdb.set_trace()
                    if node in bottom:
                        existing_nodes.append(node)
                    else:
                        shadow_locations.extend(locations(node, bottom, top_down, consumed_top_down))
                for node in existing_nodes:
                    loc = bottom.index(node)
                    candidates = []
                    for sloc in shadow_locations:
                        diff = sloc - loc
                        candidates.append((diff if diff > 0 else -diff, diff))
                    # import pdb; pdb.set_trace()
                    if len(candidates) > 1:
                        candidates.sort()
                        if candidates[0][0] == candidates[1][0]:
                            closest = candidates[randint(0, 1)][1]
                        else:
                            closest = candidates[0][1]
                    else:
                        closest = candidates[0][1]
                    soft_right[node] = closest > 0
                # import pdb; pdb.set_trace()
            continue
        
        location = []
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                location_children.append((nid, node))
        swappable_locations.append(location)

        num_children = len(location_children)
        for cid, (nid, node) in enumerate(location_children):
            if cid == 0:
                right = True
            else:
                if cid == num_children - 1:
                    right = False
                elif dependency:
                    internal_dep = False
                    for hid, h_node in lr_gen(location_children, cid):
                        if node not in dependency or h_node not in dependency:
                            # if 'VROOT' not in node and 'VROOT' not in h_node:
                            #     print(node, h_node)
                            #     import pdb; pdb.set_trace()
                            continue
                            # continue # not compactible
                        if depend_on(node, h_node):
                            internal_dep = True
                            break
                    # import pdb; pdb.set_trace()
                    if internal_dep:
                        right = nid < hid # towards head
                    else:
                        right = _positional_right(cid, num_children, factor)
                else:
                    right = _positional_right(cid, num_children, factor)
                if last_nid == nid - 1: # locality
                    if not dependency:
                        assert last_right or not right, 'incoherent' # coherent
                    if not lean_joint:
                        joint_rset.append(nid)
                    if last_right and not right: # agreeable_orients
                        if lean_joint:
                            joint_rset.append(nid)
                        group = td.children
                        if len(group) == 2:
                            new_node = p_node
                            completed_nodes.add(p_node)
                        else:
                            sub_id = 0
                            new_node = str(p_node)
                            while new_node + f'{sub_suffix}{sub_id}' in group:
                                sub_id += 1
                            new_node += f'{sub_suffix}{sub_id}'
                        if dependency and node in dependency and last_node in dependency:
                            if depend_on(last_node, node): # dep -> head
                                _new_dep(dependency, last_node, node, new_node)
                            elif depend_on(node, last_node): # head <- dep
                                _new_dep(dependency, node, last_node, new_node)
                        bottom_up[node] = new_node
                        group[new_node] = group.pop(last_node) + '.' + group.pop(node)
                        bottom_flag[last_nid] = False
                        consumed_top_down[new_node] = last_node, node
                    bottom_flag[nid] = False
                    
            right_hash[node] = last_right = right
            last_nid = nid
            last_node = node
    return right_hash, joint_rset, swappable_locations, bottom_up, soft_right

def binary_signals(bottom,
                   bottom_up,
                   top_down,
                   node2tag,
                   bottom_unary,
                   right_hash,
                   joint_rset,
                   soft_right,
                   swap_rhs_priority,
                   factor,
                   sub_prefix,
                   sub_suffix,
                   pos_prefix):
    last_right = True
    last_direc = False
    swap_distance = len(bottom)
    new_bottom = []
    right_layer = [] 
    direc_layer = []
    joint_layer = []
    label_layer = []
    for nid, node in enumerate(bottom):
        # if bottom[0] == '_s8511_500':
        #     import pdb; pdb.set_trace()
        if node in right_hash:
            right = right_hash[node]
            if last_right and not right:
                if nid in joint_rset: # [>≥]j<
                    assert new_bottom.pop() == last_node, 'bad join'
                    new_bottom.append(bottom_up.pop(node))
                else: # active swap [>≥]s<
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
            elif not last_direc and not right: # passive swap
                if swap_distance != 1: # o≥s<
                    right_layer[-1] = True
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                elif swap_rhs_priority: # >s[≥]s< restore lhs
                    right_layer[-1] = True
                    prev_node = new_bottom.pop(-2)
                    new_bottom += [node, prev_node]
                    swap_distance = 0
                else: # ??
                    new_bottom.append(node)
            else: # >
                new_bottom.append(node)
            directional = True
        else:
            if last_direc:
                if last_right: # passive swap >s≤
                    right = False
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else: # [<]o[≤≥]
                    right = soft_right.get(node, factor > 0.5)
                    new_bottom.append(node)
            else: # [≤≥]o[≤≥] TODO undetermined right
                right = soft_right.get(node, factor > 0.5)
                new_bottom.append(node)
            directional = False
        right_layer.append(right)
        direc_layer.append(directional)
        last_direc = directional
        last_right = right
        last_node = node
        swap_distance += 1
        if nid > 0:
            is_joint = nid in joint_rset
            joint_layer.append(is_joint)
        
        if sub_suffix in node:
            label_layer.append(sub_prefix + top_down[node.split('.')[0]].label)
        elif node in bottom_unary:
            label_layer.append(bottom_unary[node])
        elif node in node2tag:
            label_layer.append(pos_prefix + node2tag[node])
        else:
            label_layer.append(top_down[node].label)

    assert len(right_layer) == len(label_layer) == len(direc_layer), 'unmatched layers'
    assert len(right_layer) - 1 == len(joint_layer), 'unmatched joint layers'
    return new_bottom, right_layer, joint_layer, label_layer, direc_layer


from data.cross import has_multiple, TopDown, _read_dpenn, _read_graph, draw_str_lines, _pre_proc, gap_degree
def cross_signals(bottom, node2tag, bottom_unary, top_down, factor,
                  dependency = None,
                  aggressive = True,
                  swap_rhs_priority = None,
                  get_soft_right = False,
                  lean_joint = False,
                  sub_prefix = '_',
                  pos_prefix = '#'):
    factor = 1 - factor # range and type
    if swap_rhs_priority is None: # 
        # assert isinstance(factor, bool)
        swap_rhs_priority = factor > 0.5

    some_or_all = has_multiple if aggressive else all
    sub_suffix = '.'

    layers_of_right = []
    layers_of_joint = []
    layers_of_label = []
    layers_of_direc = []
    layers_of_swaps = []
    completed_nodes = set()
    while len(bottom) > 1:
        (right_hash, joint_rset, swappable_locations, bottom_up,
         soft_right) = binary_hash(bottom,
                                   top_down,
                                   completed_nodes,
                                   some_or_all,
                                   lean_joint,
                                   factor,
                                   sub_suffix,
                                   get_soft_right,
                                   dependency)

        (new_bottom, right_layer, joint_layer, label_layer,
         direc_layer) = binary_signals(bottom,
                                        bottom_up,
                                        top_down,
                                        node2tag,
                                        bottom_unary,
                                        right_hash,
                                        joint_rset,
                                        soft_right,
                                        swap_rhs_priority,
                                        factor,
                                        sub_prefix,
                                        sub_suffix,
                                        pos_prefix)
        if new_bottom == bottom:
            raise ValueError('should be different', bottom, top_down, bottom_unary, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc)
        bottom = new_bottom
        layers_of_right.append(right_layer)
        layers_of_joint.append(joint_layer)
        layers_of_label.append(label_layer)
        layers_of_direc.append(direc_layer)
        layers_of_swaps.append(swappable_locations)

    if top_down or len(node2tag) == len(bottom) == 1:
        layers_of_right.append([factor > 0.5])
        layers_of_direc.append([False])
        layers_of_label.append([top_down[bottom.pop()].label if top_down else node2tag[bottom.pop()]])
    return layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, layers_of_swaps

from utils.types import E_ORIF5, O_RGT, O_HEAD
from copy import deepcopy
def read_tiger_graph(graph, dep_head = None):
    bottom_info, top_down, root_id = _read_graph(graph)
    lines = draw_str_lines(bottom_info, top_down)
    word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
    bottom_tag = [node2tag[t] for t in bottom]
    gap = gap_degree(bottom, top_down, root_id)
    cnf_layers = {}
    if dep_head:
        dep_head = {node: TopDown(head, set([node])) for node, head in dep_head.items()}
        cnf_layers[O_HEAD] = cross_signals(bottom, node2tag, bottom_unary, deepcopy(top_down), 0.5, dep_head)
    for oid, cnf_factor in enumerate(E_ORIF5):
        new_top_down = deepcopy(top_down) if cnf_factor != O_RGT else top_down
        cnf_layers[cnf_factor] = cross_signals(bottom, node2tag, bottom_unary, new_top_down, (oid + 0.5) / 5)
    return word, bottom_tag, cnf_layers, gap, lines#, bottom_info, ret_top_down, root_id

def read_disco_penn(tree, dep_head = None):
    bottom_info, top_down, root_id = _read_dpenn(tree)
    lines = draw_str_lines(bottom_info, top_down)
    word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
    bottom_tag = [node2tag[t] for t in bottom]
    gap = gap_degree(bottom, top_down, root_id)
    cnf_layers = {}
    if dep_head:
        dep_head = {node: TopDown(f'n_{head}', set(['n_{node}'])) for node, head in dep_head.items()}
        cnf_layers[O_HEAD] = cross_signals(bottom, node2tag, bottom_unary, deepcopy(top_down), 0.5, dep_head)
    for oid, cnf_factor in enumerate(E_ORIF5):
        new_top_down = deepcopy(top_down) if cnf_factor != O_RGT else top_down
        cnf_layers[cnf_factor] = cross_signals(bottom, node2tag, bottom_unary, new_top_down, (oid + 0.5) / 5)
    return word, bottom_tag, cnf_layers, gap, lines

from data.cross.evalb_lcfrs import DiscoEvalb
from random import randint
class MidinTreeKeeper:
    @classmethod
    def from_graph(cls, graph):
        return cls(*_read_graph(graph))

    @classmethod
    def from_tree(cls, tree):
        return cls(*_read_dpenn(tree))

    def __init__(self, bottom_info, top_down, root_id):
        # print('\n'.join(draw_str_lines(bottom_info, top_down)))
        # g_brackets = bracketing(bottom_info, top_down, root_id)
        # self._evalb = DiscoEvalb(), set(bottom_info), g_brackets
        word, bottom, node2tag, bottom_unary = _pre_proc(bottom_info, top_down)
        self._word = word
        self._bottom_tags = [node2tag[t] for t in bottom]
        self._root_id = root_id
        self._top_down = top_down
        self._max_sub = max(len(td.children) for td in top_down.values()) if top_down else 0 # >= 2
        self._args = bottom, node2tag, bottom_unary
        self._dep = None

    def set_dependency(self, dep):
        self._dep = dep

    def binarize(self, factor):
        return self._word, self._bottom_tags, cross_signals(*self._args, deepcopy(self._top_down), factor)

    def sample(self):
        factor = randint(1, self._max_sub - 1) / self._max_sub
        return self._word, self._bottom_tags, cross_signals(*self._args, deepcopy(self._top_down), factor)

    def __str__(self):
        lines = ''
        # evalb, g_tags, g_brackets = self._evalb
        for i in range(self._max_sub - 1):
            factor = (i + 1) / self._max_sub
            args = cross_signals(*self._args, deepcopy(self._top_down), factor)[:4]
            # evalb_args = disco_tree(self._word, self._bottom_tags, *args)[:3]
            args = disco_tree(self._word, self._bottom_tags, *args, perserve_sub = True)[:2]
            lines += f'factor = {factor}\n  '
            lines += '\n  '.join(draw_str_lines(*args))
            lines += '\n'
        if self._dep:
            args = cross_signals(*self._args, deepcopy(self._top_down), 0.5, self._dep)
            # evalb_args = disco_tree(self._word, self._bottom_tags, *args)[:3]
            args = disco_tree(self._word, self._bottom_tags, *args, perserve_sub = True)[:2]
            lines += f'factor = dependency\n  '
            lines += '\n  '.join(draw_str_lines(*args))
            lines += '\n'
        #     p_tags = set(evalb_args[0])
        #     p_brackets = bracketing(*evalb_args)
        #     evalb.add(p_brackets, p_tags, g_brackets, g_tags)
        # lines += str(evalb.summary())
        # assert sum(evalb.summary()[:3]) == 300, evalb.summary()
        return lines

def targets(right_layer, joint_layer):
    targets = [1 for _ in right_layer]
    tar_len = len(targets)
    for r0id, (r0, r1, jc) in enumerate(zip(right_layer, right_layer[1:], joint_layer)):
        if r0 and not r1:
            if not jc:
                targets[r0id] += 1
                targets[r0id + 1] -= 2
                if r0id + 2 < tar_len:
                    targets[r0id + 2] += 1
    return targets

from data.cross import E_SHP, E_CMB, defaultdict, _combine, bottom_trees
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_right,
               layers_of_joint,
               layers_of_direc,
               fall_back_root_label = None,
               perserve_sub         = False):

    (NTS, bottom_len, track_nodes, terminals, non_terminals, top_down, track_fall_back,
     error_layer_id) = bottom_trees(word, bottom_tag, layers_of_label, fall_back_root_label, perserve_sub)
    non_terminal_end = NTS

    for lid, (lr, lj, ld) in enumerate(zip(layers_of_right, layers_of_joint, layers_of_direc)):
        # if not (len(lr) == len(ld) == len(lj) + 1 == len(track_nodes)):
        #     print(f'{len(lr)} == {len(ld)} == {len(lj)} + 1 == {len(lr)}')
        #     import pdb; pdb.set_trace()
        # assert len(lr) == len(ld) == len(track_nodes), f'{len(lr)} == {len(ld)} == {len(lr)} @ {lid}'
        # if lj:
        #     assert len(lj) + 1 == len(track_nodes), f'{len(lj)} + 1 == {len(lr)} @ {lid}'
        snapshot_track_nodes = track_nodes.copy()
        offset = 1
        for nid, (right, direc) in enumerate(zip(lr, ld)):
            lhs_nid = nid - offset
            if nid == 0:
                pass
            elif len(lj) == 0:
                error_layer_id = lid, E_SHP, bottom_len
                break # should be the final | unknown action in the model
            elif lj[nid - 1]: # joint
                if last_right and not right:
                    # >j<
                    lhs_node = track_nodes.pop(lhs_nid)
                    rhs_node = track_nodes.pop(lhs_nid)
                    try:
                        if not (lhs_nid < len(layers_of_label[lid + 1])):
                            import pdb; pdb.set_trace()
                            break # an error
                    except:
                        error_layer_id = lid, E_SHP, bottom_len # TODO change err_type
                        break
                    labels = layers_of_label[lid + 1][lhs_nid]
                    labels = [labels] if perserve_sub or labels[0] in '#_' else labels.split('+')
                    non_terminals[non_terminal_end] = labels.pop()
                    _combine(NTS, non_terminal_end, lhs_node, non_terminals, top_down, perserve_sub) # TODO: safe_label validate
                    _combine(NTS, non_terminal_end, rhs_node, non_terminals, top_down, perserve_sub)
                    while labels: # unary
                        non_terminal_end += 1
                        non_terminals[non_terminal_end] = labels.pop()
                        top_down[non_terminal_end] = set({non_terminal_end - 1})
                    track_nodes.insert(lhs_nid, non_terminal_end)
                    non_terminal_end += 1
                    offset += 1
            elif last_right and not right and (last_direc or direc): # cross (swap)
                # 1: >[<≤]
                # 2: [>≥]<
                # last_right and not right: # cross (swap)
                #layers_of_label[lid][nid-1:nid+1][::-1] == layers_of_label[lid+1][lhs_nid:lhs_nid+2] and\
                # 1: [≥>][<≤]
                rhs_nid = lhs_nid + 1
                # import pdb; pdb.set_trace()
                track_nodes[lhs_nid], track_nodes[rhs_nid] = track_nodes[rhs_nid], track_nodes[lhs_nid]

            last_right = right
            last_direc = direc

        # if isinstance(fall_back_root_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and snapshot_track_nodes == track_nodes and track_fall_back: # no action is taken
            if error_layer_id is None:
                error_layer_id = lid, E_CMB, bottom_len
            for pid in track_nodes:
                if pid in non_terminals and non_terminals[pid][0] in '_#':
                    non_terminals.pop(pid)
                    top_down[non_terminal_end].update(top_down.pop(pid))
                else:
                    top_down[non_terminal_end].add(pid)
            non_terminals[non_terminal_end] = fall_back_root_label
            non_terminal_end += 1
            break

    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, top_down.pop(nid))

    return terminals, top_down, non_terminal_end - 1, error_layer_id

from data.delta import get_logits
def xlogit_gen(label_layer, right_layer, direc_layer, joint_layer, current_joints, next_joints):
    for nid, (label, right, direc) in enumerate(zip(label_layer, right_layer, direc_layer)):
        is_phrase = label[0] not in '#_'
        is_joint = current_joints and current_joints[nid]
        if nid == 0:
            if not right and not direc:
                next_joints.append(False)
        elif last_right or not right:
            if joint_layer:
                jnt = joint_layer[nid - 1] # and last_right and last_direc and not right and direc
                if not jnt and (last_right and not right):# or last_right and not direc or not last_direc and not right):
                    next_joints.append(False)
            else:
                jnt = True # final layer
            next_joints.append(jnt)
        yield get_logits(right, direc, is_joint, is_phrase, False, False)
        last_right = right
        last_direc = direc
    if right and not direc:
        next_joints.append(False)

def zip_to_logit(layers_of_label, layers_of_right, layers_of_joint, layers_of_direc):
    nj = []
    cindex = []
    labels = []
    xtypes = []
    for ll, lr, lj, ld in zip(layers_of_label, layers_of_right, layers_of_joint + [None], layers_of_direc):
        if nj:
            cj = nj
            nj = []
            assert len(cj) == len(ll), f'{len(cj)} != {len(ll)}'
        else:
            cj = None
        cindex.append(len(ll))
        labels.extend(ll)
        xtypes.extend(xlogit_gen(ll, lr, ld, lj, cj, nj))
    return cindex, labels, xtypes

def zip_swaps(lsp):
    swapbl = ''
    num_layers = len(lsp)
    for lid, group_layer in enumerate(lsp):
        num_groups = len(group_layer)
        for gid, group in enumerate(group_layer):
            if gid == num_groups - 1:
                if lid == num_layers - 1:
                    swapbl += ','.join(str(x) for x in group)
                else:
                    swapbl += ','.join(str(x) for x in group) + '|'
            else:
                swapbl += ','.join(str(x) for x in group) + ';'
    return swapbl

def unzip_swaps(swapbl, offset = 0):
    lsp = []
    for layer in swapbl.split('|'):
        groups = []
        for group in layer.split(';'):
            group_idx = array([int(x) + offset for x in group.split(',')])
            group_ctn = array(group_idx)
            groups.append((group_idx, group_ctn))
        lsp.append(tuple(groups))
    return lsp

from data.trapezoid import trapezoid_to_layers
from numpy import asarray, array
from data.delta import get_rgt, get_jnt, get_dir
def unzip_xlogit(cindex, xtypes): #, the_joints, the_labels):
    xtypes = asarray(xtypes)
    rights = get_rgt(xtypes)
    joints = get_jnt(xtypes)
    direcs = get_dir(xtypes)
    joints = trapezoid_to_layers(joints, cindex, cindex, big_endian = False)
    layers_of_direc = trapezoid_to_layers(direcs, cindex, cindex, big_endian = False)
    layers_of_right = trapezoid_to_layers(rights, cindex, cindex, big_endian = False)
    layers_of_joint = []
    count = 0
    for lower_rights, lower_direcs, upper_joints in zip(layers_of_right, layers_of_direc, joints[1:]):
        joint_layer = []
        upper_joints = list(upper_joints)
        upper_joints.reverse() # pop() should be like dequeue
        for rid, (right, direc) in enumerate(zip(lower_rights, lower_direcs)):
            if rid:
                if not last_right and right and rid > 1 and not cnf_left_starter: # <≤≥>
                    joint_layer.append(False)
                else:
                    joint = upper_joints.pop()
                    joint_layer.append(joint)
                    if last_right and not right and not joint: # a swap and (last_direc or direc)
                        assert not upper_joints.pop(), (count, rid)
                if cnf_left_starter and right:
                    cnf_left_starter = False
            else:
                cnf_left_starter = not right
            last_right = right
            last_direc = direc
        if upper_joints: # factor_ender
            assert not upper_joints.pop()
        layers_of_joint.append(joint_layer)
        count += 1
              
                # if joint_layer[-1] != tj[len(joint_layer) - 1]:
                #     print(' '.join(path))
                #     import pdb; pdb.set_trace()
    # for aa, bb in zip(ld, layers_of_direc):lj, lr, ld:, lj, layers_of_label, ld, tj, tl, td
    #     assert list(bb) == aa, f'{aa} != {bb}'  
    #     c = ' '.join(str(int(x)) for x in upper_joints) + ']\nTj['
    #     a = ' '.join(str(int(x)) for x in tj) + ']\nLj['
    #     b = ' '.join(str(int(x)) for x in joint_layer) + ']\nTd['
    #     e = ' '.join(str(int(x)) for x in td) + ']\nLb['
    #     d = ' '.join((l + f'≥>'[d]) if r else (f'≤<'[d] + l) for l,d,r in zip(tl, td, lower_rights))
    #     assert len(joint_layer) == len(lower_rights) - 1, f'{len(joint_layer)} != {len(lower_rights)} - 1\n' + d
    #     if joint_layer != tj:
    #         print(' '.join(path))
    #         import pdb; pdb.set_trace()
    #     assert joint_layer == tj, '\nUj[' + c + a + b + e + d + ']\n' + f'{path}'
    return layers_of_right, layers_of_joint, layers_of_direc