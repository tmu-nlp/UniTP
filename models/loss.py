import torch
from torch.nn import functional as F

original_cross_entropy = F.cross_entropy

def cross_entropy(x_, y_, w_):
    b_, t_, c_ = x_.shape
    losses = F.cross_entropy(x_.view(-1, c_), y_.view(-1), reduction = 'none')
    if w_ is not None:
        w_ = big_endian_height_mask(t_, w_)
        losses = losses * w_.view(-1)
    return losses.sum() # TODO turn off non-train gradient tracking

def big_endian_height_mask(t_, w_):
    p_ = torch.arange(t_, device = w_.device)[None, :]
    return p_ >= w_[:, None]

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
        hinge |= ~ w
    losses[hinge] = 0
    return losses.sum()

def get_decision(argmax, logtis):
    if argmax:
        return logtis.argmax(dim = 2)
    return logtis.argmin(dim = 2)

def get_decision_with_value(score_fn, logits):
    probs = score_fn(logits)
    prob, arg = probs.topk(1)
    arg .squeeze_(dim = 2)
    prob.squeeze_(dim = 2)
    return prob, arg

def get_loss(argmax, logits, batch, *net_height_mask_key):
    if argmax:
        if len(net_height_mask_key) == 1:
            return cross_entropy(logits, batch['tag'], None)
        _, height_mask, key = net_height_mask_key
        return cross_entropy(logits, batch[key], height_mask)
        

    if len(net_height_mask_key) == 1:
        net, = net_height_mask_key
        distance = net.distance(logits, batch['tag'])
    else:
        net, height_mask, key = net_height_mask_key
        distance = net.distance(logits, batch[key]) # [b, s]
        distance *= big_endian_height_mask(distance.shape[1], height_mask)

    return distance.sum() + net.repulsion()