from random import random, randint
from data import SUB, SUP, SUBS
from data.cross import TopDown, _dep_combine, _dep_on
from data.continuous.binary import X_DIR, X_RGT, X_NEW
from numpy import asarray
from numpy.random import permutation

def has_multiple(gen):
    count = 0
    for state in gen:
        count += state
        if count > 1:
            return True
    return False

def locations(node, bottom, top_down, consumed_top_down):
    children = consumed_top_down[node] if node in consumed_top_down else top_down[node].children
    for cid in children:
        if cid in bottom:
            yield bottom.index(cid)
        else:
            yield from locations(cid, bottom, top_down, consumed_top_down)

def fill_orientation(orientation, td_children, bottom, top_down, consumed_top_down):
    nodes = []
    shadow_locations = []
    for node in td_children:
        if node in bottom:
            nodes.append(node)
        else:
            shadow_locations.extend(locations(node, bottom, top_down, consumed_top_down))
    for node in nodes:
        loc = bottom.index(node)
        candidates = []
        for sloc in shadow_locations:
            diff = sloc - loc
            candidates.append((diff if diff > 0 else -diff, diff))
        if len(candidates) > 1:
            candidates.sort()
            if candidates[0][0] == candidates[1][0]:
                closest = candidates[randint(0, 1)][1]
            else:
                closest = candidates[0][1]
        else:
            closest = candidates[0][1]
        orientation[node] = closest > 0

from utils.math_ops import lr_gen
def binary_hash(bottom,
                top_down,
                completed_nodes,
                some_or_all,
                minimal_joint,
                factor,
                nts,
                all_directional,
                dependency,
                ply_shuffle_option,
                ply_shuffle_offset):
    bottom_up = {}
    orientation = {}
    joint_rset = set()
    consumed_top_down = {}
    affinitive_groups = []
    bottom_flag = [True for _ in bottom]
    sub_top_down = {}
    for p_node, td in top_down.items():
        p_node_complete = p_node in completed_nodes
        not_enough_child = not some_or_all(node in bottom for node in td.children)
        if p_node_complete or not_enough_child:
            if all_directional and not p_node_complete and not_enough_child:
                fill_orientation(orientation, td.children, bottom, top_down, consumed_top_down)
            continue
        
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location_children.append((nid, node))
        if ply_shuffle_option > 0:
            if ply_shuffle_option > 1:
                location = asarray([n for n, _ in location_children]) + ply_shuffle_offset
            else:
                location = [n + ply_shuffle_offset for n, _ in location_children]
            affinitive_groups.append(location)

        num_children = len(location_children)
        for cid, (nid, node) in enumerate(location_children):
            if cid == 0:
                right = True
            else:
                if cid == (last_cid := num_children - 1):
                    right = False
                else:
                    if dependency:
                        internal_dep = False
                        for hid, h_node in lr_gen(location_children, cid):
                            if node not in dependency or h_node not in dependency:
                                continue
                            if _dep_on(dependency, node, h_node):
                                internal_dep = True
                                break
                        if internal_dep:
                            right = nid < hid # towards head
                    if not dependency or not internal_dep:
                        right = (cid / last_cid) < (factor[p_node] if isinstance(factor, dict) else factor)
                if last_nid == nid - 1: # locality
                    if not dependency:
                        assert last_right or not right, 'incoherent' # coherent
                    if not minimal_joint:
                        joint_rset.add(nid)
                    if last_right and not right: # agreeable_orients
                        if minimal_joint:
                            joint_rset.add(nid)
                        if len(group := td.children) == 2:
                            new_node = p_node
                            completed_nodes.add(p_node)
                        else:
                            nts -= 1; new_node = nts
                            label = top_down[p_node].label
                            if label[0] != SUB: # for efficient sub
                                label = SUB + label
                            assert new_node not in group, 'Error: overwrite existing node (with wrong nts.)'
                            group[new_node] = None
                            sub_top_down[new_node] = TopDown(label, {x: group.pop(x) for x in (last_node, node)})
                            completed_nodes.add(new_node)
                        if dependency and node in dependency and last_node in dependency:
                            if _dep_on(dependency, last_node, node): # dep -> head
                                _dep_combine(dependency, node, new_node, last_node)
                            elif _dep_on(dependency, node, last_node): # head <- dep
                                _dep_combine(dependency, last_node, new_node, node)
                        bottom_up[node] = new_node
                        bottom_flag[last_nid] = False
                        consumed_top_down[new_node] = last_node, node
                    bottom_flag[nid] = False
                    
            orientation[node] = last_right = right
            last_nid = nid
            last_node = node
    top_down.update(sub_top_down)
    return orientation, joint_rset, affinitive_groups, bottom_up, nts

def binary_signals(bottom,
                   bottom_up,
                   top_down,
                   node2tag,
                   bottom_unary,
                   orientation,
                   joint_rset,
                   swap_rhs_priority,
                   factor,
                   l2i):
    last_right = True
    last_direc = False
    swap_distance = len(bottom)
    new_bottom = []
    xtype_layer = []
    label_layer = []
    for nid, node in enumerate(bottom):
        if directional := (node in orientation):
            right = orientation[node]
            if last_right and not right:
                if nid in joint_rset: # [>≥]j<
                    assert new_bottom.pop() == last_node, 'bad join'
                    new_bottom.append(bottom_up[node])
                else: # active swap [>≥]s<
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
            elif not last_direc and not right: # passive swap
                if swap_distance != 1: # o≥s<
                    xtype_layer[-1] |= X_RGT
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else:
                    if swap_rhs_priority is None:
                        swap_rhs_priority = (factor[bottom_up(node)] if isinstance(factor, dict) else factor) < 0.5
                    if swap_rhs_priority: # >s[≥]s< restore lhs
                        print(node)
                        xtype_layer[-1] |= X_RGT
                        prev_node = new_bottom.pop(-2)
                        new_bottom += [node, prev_node]
                        swap_distance = 0
                    else: # ??
                        new_bottom.append(node)
            else: # >
                new_bottom.append(node)
        else:
            if last_direc:
                if last_right: # passive swap >s≤
                    right = False
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else:
                    right = (factor[bottom_up(node)] if isinstance(factor, dict) else factor) < 0.5
                    new_bottom.append(node)
            else: # [≤≥]o[≤≥] TODO undetermined right
                right = (factor[bottom_up(node)] if isinstance(factor, dict) else factor) < 0.5
                new_bottom.append(node)

        xtype_layer.append(right * X_RGT | X_DIR * directional)
        last_direc = directional
        last_right = right
        last_node = node
        swap_distance += 1
        
        if node in bottom_unary:
            label = bottom_unary[node]
        elif node in node2tag:
            label = SUP + node2tag[node]
        else:
            label = top_down[node].label
        label_layer.append(l2i(label) if l2i else label)

    return new_bottom, label_layer, xtype_layer


def cross_signals(bottom, node2tag, bottom_unary, top_down, factor,
                  l2i = None,
                  dependency = None,
                  aggressive = True,
                  minimal_joint = False,
                  all_directional = True,
                  swap_rhs_priority = None,
                  return_top_down = False,
                  ply_shuffle_option = 0,
                  ply_shuffle_offset = 0):

    some_or_all = has_multiple if aggressive else all
    if nts := top_down.keys() | bottom_unary.keys():
        nts = min(nts) - 1

    layers_of_joint = []
    layers_of_label = []
    layers_of_xtype = []
    layers_of_swaps = []
    completed_nodes = set()
    while len(bottom) > 1:
        (orientation, joint_layer, affinitive_groups, bottom_up,
         nts) = binary_hash(bottom,
                            top_down,
                            completed_nodes,
                            some_or_all,
                            minimal_joint,
                            factor,
                            nts,
                            all_directional,
                            dependency,
                            ply_shuffle_option,
                            ply_shuffle_offset)
        (new_bottom, label_layer,
         xtype_layer) = binary_signals(bottom,
                                       bottom_up,
                                       top_down,
                                       node2tag,
                                       bottom_unary,
                                       orientation,
                                       joint_layer,
                                       swap_rhs_priority,
                                       factor,
                                       l2i)
        if new_bottom == bottom:
            raise ValueError('should be different', bottom, top_down, bottom_unary, layers_of_label, layers_of_xtype, layers_of_joint)
        bottom = new_bottom
        layers_of_joint.append(joint_layer)
        layers_of_label.append(label_layer)
        layers_of_xtype.append(xtype_layer)
        layers_of_swaps.append(affinitive_groups)

    if top_down or len(bottom_unary) == len(bottom) == 1:
        layers_of_xtype.append([X_RGT])
        root = top_down[bottom[0]].label if top_down else bottom_unary[bottom[0]]
        layers_of_label.append([l2i(root) if l2i else root])
    if return_top_down:
        for nid, label in bottom_unary.items():
            top_down[nid] = TopDown(label, {'the bottom node': None})
        return layers_of_label, layers_of_xtype, layers_of_joint, layers_of_swaps, top_down
    return layers_of_label, layers_of_xtype, layers_of_joint, layers_of_swaps


from random import random, betavariate
from utils.types import F_LEFT, F_RIGHT, F_RANDOM, F_CNF
def extend_factor(factor, loc):
    if isinstance(factor, str):
        if factor == F_LEFT:
            factor = 0
        elif factor == F_RIGHT:
            factor = 1
        elif factor == F_RANDOM: # F_CON
            factor = {p: random() for p in loc}
    elif isinstance(factor, tuple):
        lhs, rhs = factor
        if lhs == F_CNF:
            factor = {p: random() < rhs for p in loc}
        else:
            factor = {p: betavariate(lhs, rhs) for p in loc}
    return factor


from data.cross import E_SHP, E_CMB, _combine, bottom_trees
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_xtype,
               layers_of_joint,
               fallback_label = None,
               perserve_sub   = False):

    (NTS, bottom_len, track_nodes, terminals, non_terminals, top_down, track_fall_back,
     error_layer_id) = bottom_trees(word, bottom_tag, layers_of_label, fallback_label, perserve_sub)

    for lid, (lx, lj) in enumerate(zip(layers_of_xtype, layers_of_joint)):
        offset = 1
        snapshot_track_nodes = track_nodes.copy()
        for nid, xtype in enumerate(lx):
            right = xtype & X_RGT
            direc = xtype & X_DIR
            lhs_nid = nid - offset
            if nid == 0:
                if not isinstance(lj, set): lj = set(lj)
            elif nid in lj: # joint
                if last_right and not right:
                    # >j<
                    lhs_node = track_nodes.pop(lhs_nid)
                    rhs_node = track_nodes[lhs_nid]
                    if fallback_label and (lid + 1 == len(layers_of_label)):
                        break
                    labels = layers_of_label[lid + 1][lhs_nid]
                    labels = [labels] if perserve_sub or labels[0] in SUBS else labels.split('+')
                    non_terminals[NTS] = labels.pop()
                    _combine(NTS, lhs_node, non_terminals, top_down, perserve_sub)
                    _combine(NTS, rhs_node, non_terminals, top_down, perserve_sub)
                    while labels: # unary
                        NTS -= 1
                        non_terminals[NTS] = labels.pop()
                        top_down[NTS] = set({NTS + 1})
                    track_nodes[lhs_nid] = NTS
                    offset += 1
                    NTS -= 1
            elif last_right and not right and (last_direc or direc): # cross (swap)
                # 1: >[<≤]
                # 2: [>≥]<
                # last_right and not right: # cross (swap)
                #layers_of_label[lid][nid-1:nid+1][::-1] == layers_of_label[lid+1][lhs_nid:lhs_nid+2] and\
                # 1: [≥>][<≤]
                rhs_nid = lhs_nid + 1
                track_nodes[lhs_nid], track_nodes[rhs_nid] = track_nodes[rhs_nid], track_nodes[lhs_nid]

            last_right = right
            last_direc = direc

        if len(track_nodes) > 1 and snapshot_track_nodes == track_nodes and track_fall_back: # no action is taken
            if error_layer_id is None:
                error_layer_id = lid, bottom_len, E_CMB
            break

    if error_layer_id or not top_down or len(track_nodes) > 1:
        if error_layer_id is None:
            error_layer_id = lid, bottom_len, E_SHP
        for pid in track_nodes:
            if pid in non_terminals and non_terminals[pid][0] in SUBS:
                non_terminals.pop(pid)
                top_down[NTS].update(top_down.pop(pid))
            else:
                top_down[NTS].add(pid)
        non_terminals[NTS] = fallback_label
        NTS -= 1

    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, {x: None for x in top_down.pop(nid)})
    top_down[0] = top_down.pop(NTS + 1)
    return terminals, top_down, error_layer_id