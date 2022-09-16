from random import random, randint
from data.cross import TopDown, _dep_combine, _dep_on

def has_multiple(gen):
    count = 0
    for state in gen:
        count += state
        if count > 1:
            return True
    return False

def _positional_right(cid, num_children, factor):
    position = 1 - (cid + 0.5) / num_children
    if position == factor == 0.5:
        right = random() > 0.5
    else:
        right = position > factor # T[TfF]F
    return right

def locations(node, bottom, top_down, consumed_top_down):
    children = consumed_top_down[node] if node in consumed_top_down else top_down[node].children
    for cid in children:
        if cid in bottom:
            yield bottom.index(cid)
        else:
            yield from locations(cid, bottom, top_down, consumed_top_down)

def _add_right(right_hash, td_children, bottom, top_down, consumed_top_down):
    existing_nodes = []
    shadow_locations = []
    for node in td_children:
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
        right_hash[node] = closest > 0

from utils.math_ops import lr_gen
def binary_hash(bottom,
                top_down,
                completed_nodes,
                some_or_all,
                lean_joint,
                factor,
                nts,
                remove_undirec,
                dependency,
                sub_prefix):
    bottom_up = {}
    right_hash = {}
    joint_rset = []
    consumed_top_down = {}
    swappable_groups = []
    bottom_flag = [True for _ in bottom]
    sub_top_down = {}
    for p_node, td in top_down.items():
        p_node_complete = p_node in completed_nodes
        not_enough_child = not some_or_all(node in bottom for node in td.children)
        # if p_node == 'n_506':
        #     import pdb; pdb.set_trace()
        if p_node_complete or not_enough_child:
            if remove_undirec and not p_node_complete and not_enough_child:
                _add_right(right_hash, td.children, bottom, top_down, consumed_top_down)
            continue
        
        location = []
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location.append(nid)
                location_children.append((nid, node))
        swappable_groups.append(location)

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
                        if _dep_on(dependency, node, h_node):
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
                        if len(group := td.children) == 2:
                            new_node = p_node
                            completed_nodes.add(p_node)
                        else:
                            nts -= 1; new_node = nts
                            label = top_down[p_node].label
                            if label[0] != sub_prefix: # for efficient sub
                                label = sub_prefix + label
                            assert new_node not in group, 'Error: overwrite existing node (with wrong nts.)'
                            group[new_node] = None
                            sub_top_down[new_node] = TopDown(label, {x: group.pop(x) for x in (last_node, node)})
                            completed_nodes.add(new_node)
                            # new_node = p_node.rfind(sub_suffix)
                            # new_node = p_node if new_node < 0 else p_node[:new_node]
                            # new_node += _new_sub_id(group, sub_suffix)
                            # sub_id = 0
                            # while new_node + f'{sub_suffix}{sub_id}' in group:
                            #     sub_id += 1
                            # new_node += f'{sub_suffix}{sub_id}'
                        if dependency and node in dependency and last_node in dependency:
                            if _dep_on(dependency, last_node, node): # dep -> head
                                _dep_combine(dependency, node, new_node, last_node)
                            elif _dep_on(dependency, node, last_node): # head <- dep
                                _dep_combine(dependency, last_node, new_node, node)
                        bottom_up[node] = new_node
                        bottom_flag[last_nid] = False
                        consumed_top_down[new_node] = last_node, node
                    bottom_flag[nid] = False
                    
            right_hash[node] = last_right = right
            last_nid = nid
            last_node = node
    top_down.update(sub_top_down)
    return right_hash, joint_rset, swappable_groups, bottom_up, nts

def binary_signals(bottom,
                   bottom_up,
                   top_down,
                   node2tag,
                   bottom_unary,
                   right_hash,
                   joint_rset,
                   swap_rhs_priority,
                   factor,
                   pos_prefix,
                   l2i):
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
                else:
                    right = factor > 0.5
                    new_bottom.append(node)
            else: # [≤≥]o[≤≥] TODO undetermined right
                right = factor > 0.5
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
        
        if node in bottom_unary:
            label = bottom_unary[node]
        elif node in node2tag:
            label = pos_prefix + node2tag[node]
        else:
            label = top_down[node].label
        label_layer.append(l2i(label) if l2i else label)

    assert len(right_layer) == len(label_layer) == len(direc_layer), 'unmatched layers'
    assert len(right_layer) - 1 == len(joint_layer), 'unmatched joint layers'
    return new_bottom, right_layer, joint_layer, label_layer, direc_layer


def cross_signals(bottom, node2tag, bottom_unary, top_down, factor,
                  dependency = None,
                  aggressive = True,
                  swap_rhs_priority = None,
                  remove_undirec = True,
                  lean_joint = False,
                  sub_prefix = '_',
                  pos_prefix = '#',
                  l2i = None):
    factor = 1 - factor # range and type
    if swap_rhs_priority is None: # 
        # assert isinstance(factor, bool)
        swap_rhs_priority = factor > 0.5

    some_or_all = has_multiple if aggressive else all
    if nts := top_down.keys() | bottom_unary.keys():
        nts = min(nts) - 1

    layers_of_right = []
    layers_of_joint = []
    layers_of_label = []
    layers_of_direc = []
    layers_of_swaps = []
    completed_nodes = set()
    while len(bottom) > 1:
        (right_hash, joint_rset, swappable_groups, bottom_up,
         nts) = binary_hash(bottom,
                            top_down,
                            completed_nodes,
                            some_or_all,
                            lean_joint,
                            factor,
                            nts,
                            remove_undirec,
                            dependency,
                            sub_prefix)
        (new_bottom, right_layer, joint_layer, label_layer,
         direc_layer) = binary_signals(bottom,
                                       bottom_up,
                                       top_down,
                                       node2tag,
                                       bottom_unary,
                                       right_hash,
                                       joint_rset,
                                       swap_rhs_priority,
                                       factor,
                                       pos_prefix,
                                       l2i)
        if new_bottom == bottom:
            raise ValueError('should be different', bottom, top_down, bottom_unary, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc)
        bottom = new_bottom
        layers_of_right.append(right_layer)
        layers_of_joint.append(joint_layer)
        layers_of_label.append(label_layer)
        layers_of_direc.append(direc_layer)
        layers_of_swaps.append(swappable_groups)

    if top_down or len(bottom_unary) == len(bottom) == 1:
        layers_of_right.append([factor > 0.5])
        layers_of_direc.append([False])
        root = top_down[bottom[0]].label if top_down else bottom_unary[bottom[0]]
        layers_of_label.append([l2i(root) if l2i else root])
    return layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, layers_of_swaps


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

from data.cross import E_SHP, E_CMB, _combine, bottom_trees
def disco_tree(word, bottom_tag, 
               layers_of_label,
               layers_of_right,
               layers_of_joint,
               layers_of_direc,
               fallback_label = None,
               perserve_sub   = False):

    (NTS, bottom_len, track_nodes, terminals, non_terminals, top_down, track_fall_back,
     error_layer_id) = bottom_trees(word, bottom_tag, layers_of_label, fallback_label, perserve_sub)

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
                    non_terminals[NTS] = labels.pop()
                    _combine(NTS, lhs_node, non_terminals, top_down, perserve_sub) # TODO: safe_label validate
                    _combine(NTS, rhs_node, non_terminals, top_down, perserve_sub)
                    while labels: # unary
                        NTS -= 1
                        non_terminals[NTS] = labels.pop()
                        top_down[NTS] = set({NTS + 1})
                    track_nodes.insert(lhs_nid, NTS)
                    NTS -= 1
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

        # if isinstance(fallback_label, str) and len(word) > 2:
        #     import pdb; pdb.set_trace()
        if len(track_nodes) > 1 and snapshot_track_nodes == track_nodes and track_fall_back: # no action is taken
            if error_layer_id is None:
                error_layer_id = lid, E_CMB, bottom_len
            for pid in track_nodes:
                if pid in non_terminals and non_terminals[pid][0] in '_#':
                    non_terminals.pop(pid)
                    top_down[NTS].update(top_down.pop(pid))
                else:
                    top_down[NTS].add(pid)
            non_terminals[NTS] = fallback_label
            NTS -= 1
            break

    for nid, label in non_terminals.items():
        top_down[nid] = TopDown(label, {x: None for x in top_down.pop(nid)})
    top_down[0] = top_down.pop(NTS + 1)

    return terminals, top_down, error_layer_id

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

from utils.str_ops import zip_to_str, unzip_from_str
def zip_group(group):
    return zip_to_str(group, ',', str)
def zip_layer(layer):
    return zip_to_str(layer, ';', zip_group)
def zip_swaps(layers):
    return zip_to_str(layers, '|', zip_layer)

from numpy import asarray, fromiter
def unzip_group(group):
    return fromiter((int(i) for i in group.split(',')), int)
unzip_swaps = lambda layers: unzip_from_str(layers, '|', tuple, ';', tuple, unzip_group)
def unzip_group_and_double(group):
    origin = fromiter((int(i) for i in group.split(',')), int) + 1
    return origin, origin.copy()
unzip_and_double_swaps_p1 = lambda layers: unzip_from_str(layers, '|', tuple, ';', tuple, unzip_group_and_double)

def double_asarray(group):
    origin = asarray(group) + 1
    return origin, origin.copy()
def double_layer(layer):
    return tuple(double_asarray(group) for group in layer)
def double_swaps_p1(layers):
    return tuple(double_layer(layer) for layer in layers)

from data.continuous.binary.trapezoid import trapezoid_to_layers
from data.continuous.binary import X_RGT, X_NEW, X_DIR
# from data.delta import get_rgt, get_jnt, get_dir
def unzip_xlogit(cindex, xtypes): #, the_joints, the_labels):
    xtypes = asarray(xtypes)
    rights = X_RGT & xtypes
    joints = X_NEW & xtypes
    direcs = X_DIR & xtypes
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

from data.mp import DM
from data.cross.evalb_lcfrs import export_string

class BxDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, bid_offset, segments, *data_gen):
        for s_seq_len, s_token, s_tag, s_label, s_right, s_joint, s_direc in zip(*data_gen):
            layers_of_label = []
            layers_of_right = []
            layers_of_joint = []
            layers_of_direc = []
            jnt_start = 0
            rgt_start = 0
            for s_size, s_len in zip(segments, s_seq_len):
                label_layer = tuple(i2l(i) for i in s_label[rgt_start + 1: rgt_start + s_len + 1])
                layers_of_label.append(label_layer)
                layers_of_joint.append(s_joint[jnt_start + 1: jnt_start + s_len])
                layers_of_right.append(s_right[rgt_start + 1: rgt_start + s_len + 1])
                layers_of_direc.append(s_direc[rgt_start + 1: rgt_start + s_len + 1])
                rgt_start += s_size
                jnt_start += s_size - 1
                if s_len == 1:
                    break
            bottom_end = s_seq_len[0] + 1
            tags  = tuple(i2t[i] for i in   s_tag[1:bottom_end])
            words = tuple(i2w[i] for i in s_token[1:bottom_end])
            bt, td, _ = disco_tree(words, tags, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc, 'VROOT')
            yield export_string(bid_offset, bt, td)
            bid_offset += 1

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        # import pdb; pdb.set_trace()
        bid_offset, segment = args[:2]
        start = seg_id * seg_size
        if start < batch_size:
            return (bid_offset + start, segment) + tuple(x[start: (seg_id + 1) * seg_size] for x in args[2:])