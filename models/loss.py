import torch
from torch.nn import functional as F

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