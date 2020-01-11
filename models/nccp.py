import torch
from torch import nn, Tensor
from models.backend import Stem, activation_type

from utils.types import hidden_dim, frac_4, BaseType
def valid_codebook(name):
    if name.startswith('codebook'):
        if '|' in name:
            bar = name.index('|') + 1
            try:
                bar = float(name[bar:])
            except:
                return False
        return bar >= 0
    return False
    
logit_type = BaseType('affine', default_set = ('affine', 'linear', 'codebook'), validator = valid_codebook)
multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

from models.backend import stem_config
penn_tree_config = dict(orient_layer    = stem_config,
                        tag_label_layer = multi_class)
from models.utils import GaussianCodebook
from models.loss import cross_entropy, height_mask as make_

class BaseRnnTree(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 orient_layer,
                 tag_label_layer):
        super().__init__()

        self._stem_layer = Stem(model_dim, **orient_layer)

        hidden_dim = tag_label_layer['hidden_dim']
        activation = tag_label_layer['activation']
        self._shared_layer = nn.Linear(model_dim, hidden_dim)
        self._dp_layer = nn.Dropout(tag_label_layer['drop_out'])

        logit_type = tag_label_layer['logit_type']
        if logit_type in ('affine', 'linear'):
            Net = lambda i_size, o_size: nn.Linear(i_size, o_size, bias = logit_type == 'affine')
            argmax = True
            self._score = nn.Softmax(dim = 2)
            self._activation = activation()
        elif logit_type.startswith('codebook'):
            Net = GaussianCodebook
            argmax = False
            self._score = nn.Softmin(dim = 2)
            if '|' in logit_type:
                bar = logit_type.index('|') + 1
                self._repulsion = float(logit_type[bar:])
            else:
                self._repulsion = 0
            
        self._tag_layer   = Net(hidden_dim, num_tags)
        self._label_layer = Net(hidden_dim, num_labels)
        self._logit_max = argmax

    def forward(self,
                base_inputs,
                bottom_existence,
                **kw_args):

        (layers_of_base, layers_of_orient, layers_of_existence,
         trapezoid_info) = self._stem_layer(bottom_existence,
                                            base_inputs, # dynamic can be none
                                            **kw_args)

        layers_of_hidden = self._shared_layer(layers_of_base)
        layers_of_hidden = self._dp_layer(layers_of_hidden)
        if self._logit_max:
            layers_of_hidden = self._activation(layers_of_hidden)
        _, batch_len, _ = base_inputs.shape
        tags    = self._tag_layer(layers_of_hidden[:, -batch_len:])
        labels  = self._label_layer(layers_of_hidden)

        return layers_of_base, layers_of_existence, layers_of_orient, tags, labels, trapezoid_info

    def get_decision(self, logtis):
        if self._logit_max:
            return logtis.argmax(dim = 2)
        return logtis.argmin(dim = 2)

    def get_decision_with_value(self, logits):
        probs = self._score(logits)
        prob, arg = probs.topk(1)
        arg .squeeze_(dim = 2)
        prob.squeeze_(dim = 2)
        return prob, arg

    def get_loss(self, logits, batch, height_mask):
        if self._logit_max:
            if height_mask is None:
                return cross_entropy(logits, batch['tag'], None)
            return cross_entropy(logits, batch['label'], height_mask)
        
        if height_mask is None:
            net = self._tag_layer
            distance = net.distance(logits, batch['tag'])
        else:
            net = self._label_layer
            distance = net.distance(logits, batch['label']) # [b, s]
            distance *= make_(distance.shape[1], height_mask)
        loss = distance.sum()

        if self._repulsion > 0:
            loss += net.repulsion(self._repulsion)
        return loss