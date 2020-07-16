from collections import namedtuple, defaultdict, Counter
TopDown = namedtuple('TopDown', 'label, children')

def has_multiple(gen):
    count = 0
    for state in gen:
        count += state
        if count > 1:
            return True
    return False

# def list_swap(bottom, lhs, rhs):
#     bottom[lhs], bottom[rhs] = bottom[rhs], bottom[lhs]
from utils.shell_io import byte_style

def _read_graph(graph, unary_join_mark = '+'):
    top_down = {}
    unary = {}
    single_attachment = set()
    for nt in graph[1]:
        p_node = nt.get('id')
        label = nt.get('cat')
        children = {}

        for edge in nt:
            if edge.tag == 'secedge':
                continue
            node = edge.get('idref')
            children[node] = edge.get('label')
            assert node not in single_attachment, 'multi-attachment'
            single_attachment.add(node)
        
        if len(children) == 1:
            unary[node] = (label, p_node)
        else:
            top_down[p_node] = TopDown(label, children)

    word = []
    bottom = []
    node2tag = {}
    bottom_unary = {}
    for t in graph[0]:
        node = t.get('id')
        collapsed_label = ''
        while node in unary:
            label, node = unary.pop(node)
            collapsed_label += unary_join_mark + label
        if collapsed_label:
            bottom_unary[node] = collapsed_label[1:]
        node2tag[node] = t.get('pos')
        bottom.append(node)
        word.append(t.get('word'))

    for node, (label, p_node) in unary.items():
        td_label, children = top_down.pop(node)
        top_down[p_node] = TopDown(label + unary_join_mark + td_label, children)


    return word, bottom, top_down, node2tag, bottom_unary

def _layer_base(bottom, top_down, completed_nodes, some_or_all, cnf_right, sub_suffix):
    bottom_up = {}
    right_hash = {}
    joint_rset = []
    bottom_flag = [True for _ in bottom]
    for p_node, td in top_down.items():
        if p_node in completed_nodes or not some_or_all(node in bottom for node in td.children):
            continue
        
        location_children = []
        for nid, (node, flag) in enumerate(zip(bottom, bottom_flag)):
            if flag and node in td.children:
                location_children.append((nid, node))

        num_children = len(location_children)
        for cid, (nid, node) in enumerate(location_children):
            if cid == 0:
                right = True
            else:
                right = False if cid == num_children - 1 else cnf_right
                if last_nid == nid - 1: # locality
                    joint_rset.append(nid)
                    assert last_right or not right # coherent
                    if last_right and not right: # agreeable_orients
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
                        bottom_up[node] = new_node
                        group[new_node] = group.pop(last_node) + '.' + group.pop(node)
                        bottom_flag[last_nid] = False
                    bottom_flag[nid] = False
                    
            right_hash[node] = last_right = right
            last_nid = nid
            last_node = node
    return right_hash, joint_rset, bottom_up

def _layer_output(bottom,
                  bottom_up,
                  top_down,
                  node2tag,
                  bottom_unary,
                  right_hash,
                  joint_rset,
                  swap_right_priority,
                  cnf_right,
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
                if nid in joint_rset:
                    assert new_bottom.pop() == last_node
                    new_bottom.append(bottom_up.pop(node))
                else: # active swap
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
            elif not last_direc and not right: # passive swap
                if swap_distance != 1:
                    right_layer[-1] = True
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                    # print(byte_style(' @', '1'), end = '')
                elif swap_right_priority:
                    prev_node = new_bottom.pop(-2)
                    new_bottom += [node, prev_node]
                    swap_distance = 0
                    # print(byte_style('rhs@', '2'), end = '')
                else:
                    new_bottom.append(node)
                    # print(byte_style('@lhs', '3'), end = '')
            else:
                new_bottom.append(node)
            directional = True
        else:
            if last_direc:
                right = cnf_right
                if last_right and not cnf_right: # passive swap
                    last_node = new_bottom.pop()
                    new_bottom += [node, last_node]
                    swap_distance = 0
                else:
                    new_bottom.append(node)
            else:
                right = last_right
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
            # print(' x'[is_joint], end = '')
        
        if sub_suffix in node:
            label_layer.append(sub_prefix + top_down[node.split('.')[0]].label)
        elif node in bottom_unary:
            label_layer.append(bottom_unary[node])
        elif node in node2tag:
            label_layer.append(pos_prefix + node2tag[node])
        else:
            label_layer.append(top_down[node].label)

    #     _node = node.split('_')[-1]
    #     if node[0] == sub_prefix:
    #         _node = sub_prefix + _node
    #     if right:
    #         print(_node.rjust(5) + '≥>'[directional], end = '')
    #     else:
    #         print('≤<'[directional] + _node.ljust(5), end = '')
    # print()
    # target_layer = targets(right_layer, joint_layer)
    # for t in target_layer:
    #     print(str(t).center(6), end = ' ')
    # print()
    assert len(right_layer) == len(label_layer) == len(direc_layer)
    assert len(right_layer) - 1 == len(joint_layer)
    return new_bottom, right_layer, joint_layer, label_layer, direc_layer


def cross_signals(graph, cnf_right,
                  aggressive = True,
                  swap_right_priority = True,
                  sub_prefix = '_',
                  pos_prefix = '#'):
    word, bottom, top_down, node2tag, bottom_unary = _read_graph(graph[0])
    some_or_all = has_multiple if aggressive else all
    bottom_tag = [node2tag[t] for t in bottom]
    sub_suffix = '.'

    layers_of_right = []
    layers_of_joint = []
    layers_of_label = []
    layers_of_direc = []
    completed_nodes = set()
    while len(bottom) > 1:
        (right_hash, joint_rset,
         bottom_up) = _layer_base(bottom,
                                  top_down,
                                  completed_nodes,
                                  some_or_all,
                                  cnf_right,
                                  sub_suffix)

        (new_bottom, right_layer, joint_layer, label_layer,
         direc_layer) = _layer_output(bottom,
                                      bottom_up,
                                      top_down,
                                      node2tag,
                                      bottom_unary,
                                      right_hash,
                                      joint_rset,
                                      swap_right_priority,
                                      cnf_right,
                                      sub_prefix,
                                      sub_suffix,
                                      pos_prefix)
        # if new_bottom == bottom:
        #     import pdb; pdb.set_trace()
        assert new_bottom != bottom, 'should be different'
        bottom = new_bottom
        layers_of_right.append(right_layer)
        layers_of_joint.append(joint_layer)
        layers_of_label.append(label_layer)
        layers_of_direc.append(direc_layer)

    if top_down:
        layers_of_right.append([cnf_right])
        layers_of_direc.append([False])
        layers_of_label.append([top_down[bottom.pop()].label])
    return word, bottom_tag, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc

def targets(right_layer, joint_layer):
    targets = [1 for _ in right_layer]
    tar_len = len(targets)
    for r0id, (r0, r1, jc) in enumerate(zip(right_layer, right_layer[1:], joint_layer)):
        if r0 and not r1:
            if jc:
                targets[r0id + 1] = 0
            else:
                targets[r0id] += 1
                targets[r0id + 1] -= 2
                if r0id + 2 < tar_len:
                    targets[r0id + 2] += 1
    return targets

def disco_tree(word, bottom_tag, layers_of_label, layers_of_right, layers_of_joint, layers_of_direc):
    track_nodes = []
    terminals = []
    non_terminals = {}
    top_down = defaultdict(set)
    NTS = 500
    for tid, wd_tg in enumerate(zip(word, bottom_tag)):
        terminals.append(wd_tg)
        if layers_of_label[0][tid][0] not in '#_':
            bottom_unary = layers_of_label[0][tid].split('+')
            last_node = tid
            while bottom_unary:
                non_terminals[NTS] = bottom_unary.pop()
                top_down[NTS] = set({last_node})
                last_node = NTS
                NTS += 1
            track_nodes.append(NTS - 1)
        else:
            track_nodes.append(tid)
            
    non_terminal_start = NTS
    def _combine(child_node, parent_node, non_terminals, top_down):
        if child_node < NTS or non_terminals[child_node][0] not in '#_':
            top_down[parent_node].add(child_node)
            safe_label = None
        else:
            top_down[parent_node] |= top_down.pop(child_node)
            safe_label = non_terminals.pop(child_node)[1:]
            safe_label = non_terminals[parent_node].endswith(safe_label)
        return safe_label

    for lid, (lr, lj, ld) in enumerate(zip(layers_of_right, layers_of_joint, layers_of_direc)):
        for nid, (right, direc) in enumerate(zip(lr, ld)):
            if nid == 0:
                offset = 1
            elif lj[nid - 1]: # joint
                if last_right and not right:
                    lhs_nid = nid - offset
                    lhs_node = track_nodes.pop(lhs_nid)
                    rhs_node = track_nodes.pop(lhs_nid)
                    labels = layers_of_label[lid + 1][lhs_nid]
                    labels = [labels] if labels[0] in '#_' else labels.split('+')
                    non_terminals[non_terminal_start] = labels.pop()
                    _combine(lhs_node, non_terminal_start, non_terminals, top_down) # TODO: safe_label validate
                    _combine(rhs_node, non_terminal_start, non_terminals, top_down)
                    while labels:
                        non_terminal_start += 1
                        non_terminals[non_terminal_start] = labels.pop()
                        top_down[non_terminal_start] = set({non_terminal_start - 1})
                    track_nodes.insert(lhs_nid, non_terminal_start)
                    non_terminal_start += 1
                    offset += 1
            elif last_direc and last_right and (not direc or direc and not right) or not last_direc and direc and not right: # cross
                lhs_nid = nid - offset
                rhs_nid = lhs_nid + 1
                track_nodes[lhs_nid], track_nodes[rhs_nid] = track_nodes[rhs_nid], track_nodes[lhs_nid]

            last_right = right
            last_direc = direc

    return top_down, terminals, non_terminals

from data.delta import get_logits as _get_logits
def xlogit_gen(label_layer, right_layer, direc_layer, joint_layer, current_joints, next_joints):
    for nid, (label, right, direc) in enumerate(zip(label_layer, right_layer, direc_layer)):
        is_phrase = label[0] not in '#_'
        is_joint = current_joints and current_joints[nid]
        if nid and (last_right or not right):
            if joint_layer:
                jnt = joint_layer[nid - 1] # and last_right and last_direc and not right and direc
                if not jnt and (last_right and not right):# or last_right and not direc or not last_direc and not right):
                    next_joints.append(False)
            else:
                jnt = True # final layer
            next_joints.append(jnt)
        yield _get_logits(right, direc, is_joint, is_phrase, False, False)
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

from data.trapezoid import trapezoid_to_layers
from numpy import asarray
from data.delta import get_rgt, get_jnt, get_dir
def unzip_xlogit(cindex, xtypes):
    xtypes = asarray(xtypes)
    rights = get_rgt(xtypes)
    joints = get_jnt(xtypes)
    direcs = get_dir(xtypes)
    joints = trapezoid_to_layers(joints, cindex, cindex, big_endian = False)
    layers_of_direc = trapezoid_to_layers(direcs, cindex, cindex, big_endian = False)
    layers_of_right = trapezoid_to_layers(rights, cindex, cindex, big_endian = False)
    layers_of_joint = []
    for lower_rights, upper_joints in zip(layers_of_right, joints[1:]):
        joint_layer = []
        upper_joints = list(upper_joints)
        upper_joints.reverse() # pop() should be like dequeue
        for rid, right in enumerate(lower_rights):
            if rid:
                if not last_right and right:
                    joint_layer.append(False)
                else:
                    joint = upper_joints.pop()
                    joint_layer.append(joint)
                    if last_right and not right and not joint: # a swap
                        assert not upper_joints.pop()
            last_right = right
        if upper_joints:
            assert not upper_joints.pop()
        layers_of_joint.append(joint_layer)
              
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