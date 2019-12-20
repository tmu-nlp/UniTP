import torch
from torch import nn, Tensor
from models.backend import Stem, activation_type

from utils.types import hidden_dim, frac_4
multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   drop_out   = frac_4)

from models.backend import stem_config
penn_tree_config = dict(orient_layer = stem_config,
                        tag_label_layer = multi_class)

class BasePennTree(nn.Module):
    def __init__(self,
                 unit_dim,
                 num_tags,
                 num_labels,
                 orient_layer,
                 tag_label_layer):
        super().__init__()

        self._stem_layer = Stem(unit_dim, **orient_layer)

        hidden_dim = tag_label_layer['hidden_dim']
        activation = tag_label_layer['activation']
        self._shared_layer = nn.Linear(unit_dim, hidden_dim)
        self._activation = activation()
        self._dp_layer = nn.Dropout(tag_label_layer['drop_out'])

        self._tag_layer   = nn.Linear(hidden_dim, num_tags)
        self._label_layer = nn.Linear(hidden_dim, num_labels)

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
        layers_of_hidden = self._activation(layers_of_hidden)
        _, batch_len, _ = base_inputs.shape
        tags    = self._tag_layer(layers_of_hidden[:, -batch_len:])
        labels  = self._label_layer(layers_of_hidden)

        return layers_of_base, layers_of_existence, layers_of_orient, tags, labels, trapezoid_info