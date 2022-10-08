import torch
from torch.nn import functional as F
from utils.math_ops import s_index

def cross_entropy(x_, y_, w_ = None):
    c_ = x_.shape[-1]
    losses = F.cross_entropy(x_.view(-1, c_), y_.view(-1), reduction = 'none')
    if w_ is not None:
        losses = losses * w_.view(-1)
    return losses.sum() # TODO turn off non-train gradient tracking

def binary_cross_entropy(x, y, w):
    losses = F.binary_cross_entropy(x, y.type(x.dtype), reduction = 'none')
    if w is not None:
        losses = losses * w
    return losses.sum()

def hinge_loss(x, y, w):
    ones = torch.ones_like(x)
    y = torch.where(y, ones, -ones)
    losses = 1 - (x * y)
    hinge = losses < 0
    if w is not None:
        if w.is_floating_point():
            losses = losses * w
        else:
            hinge |= ~ w
    losses[hinge] = 0
    return losses.sum()

def get_decision(argmax, logits):
    if argmax:
        return logits.argmax(dim = 2)
    return logits.argmin(dim = 2)

def get_decision_with_value(score_fn, logits):
    prob, arg = sorted_decisions_with_values(score_fn, 1, logits)
    arg .squeeze_(dim = 2)
    prob.squeeze_(dim = 2)
    return prob, arg

def sorted_decisions(argmax, topk, logits):
    return logits.topk(topk, largest = argmax)[1]

def sorted_decisions_with_values(score_fn, topk, logits):
    return score_fn(logits).topk(topk)

def get_loss(net, logits, gold, weight = None):
    if net is None:
        return cross_entropy(logits, gold, weight)

    distance = net.distance(logits, gold)
    if weight is not None:
        distance = distance * weight

    return distance.sum() + net.repulsion()

def get_label_height_mask(batch, key = 'label', extra_pad = 0):
    label = batch[key]
    s_dim = torch.arange(label.shape[1], device = label.device)
    if (segment := batch.get('segment')) is None:
        length = batch['length']
        if (offset := batch.get('offset')) is None:
            seq = s_index(length)
        else:
            seq = s_index(length + offset) - s_index(offset)
    else: # trapezoid
        segment = torch.as_tensor(segment, device = label.device)
        seg = segment.max(dim = 0).values
        if extra_pad: seg += extra_pad
        seq = (seg[None] * (segment > 0)).sum(dim = 1)
    return s_dim[None] < seq[:, None]