from models.backend import ParsingOutputLayer
from torch import nn


class SentimentOnSyntactic(ParsingOutputLayer):
    def __init__(self,
                 stem_fn,
                 model_dim,
                 num_tags,
                 num_labels,
                 num_polars,
                 stem_layer,
                 tag_label_layer,
                 polar_layer,
                 **kwargs_forwarding):
        super().__init__(stem_fn,
                         model_dim,
                         num_tags,
                         num_labels,
                         stem_layer,
                         tag_label_layer,
                         **kwargs_forwarding)
        hidden_dim = polar_layer['hidden_dim']
        if hidden_dim:
            self._shared_layer = nn.Linear(model_dim, hidden_dim)
            self._dp_layer = nn.Dropout(tag_label_layer['drop_out'])

            self._polar_layer = Net(hidden_dim, num_labels) if num_polars else None
        

    def forward(self, base_inputs, bottom_existence, ignore_logits=False, key=None, tag_layer=0, **kw_args):
        return super().forward(base_inputs, bottom_existence, ignore_logits, key, tag_layer, **kw_args)