from experiments.t_lstm_parse.model import PennRnnTree, penn_tree_config
from models.backend import activation_type, nn
from utils.types import true_type, hidden_dim, frac_4

sentiment_layer = dict(indie_from_parsing_hidden = true_type,
                       indie_hidden_dim = hidden_dim,
                       indie_activation = activation_type,
                       indie_drop_out   = frac_4)

model_type = penn_tree_config.copy()
model_type['sentiment_layer']  = sentiment_layer

class StanRnnTree(PennRnnTree):
    def __init__(self,
                 num_sentiment,
                 sentiment_layer,
                 *base_args,
                 **base_kwargs):
        super().__init__(*base_args, **base_kwargs)

        indie_from_parsing_hidden = sentiment_layer['indie_from_parsing_hidden']
        indie_hidden_dim = sentiment_layer['indie_hidden_dim']
        indie_activation = sentiment_layer['indie_activation']
        indie_drop_out   = sentiment_layer['indie_drop_out']
        
        if indie_from_parsing_hidden:
            if indie_hidden_dim:
                self._indie_hidden = a = nn.Linear(self.model_dim, indie_hidden_dim)
                self._activation = b =indie_activation()
                self._dp_layer = c = nn.Dropout(indie_drop_out)
                self._sentiment = d = nn.Linear(indie_hidden_dim, num_sentiment)
                def get_sentiment(input):
                    return d(c(b(a(input))))
            else:
                self._sentiment = get_sentiment = nn.Linear(self.model_dim, num_sentiment)
        else:
            self._sentiment = get_sentiment = nn.Linear(base_kwargs['tag_label_layer']['hiddin_dim'], num_sentiment)
        self._get_sentiment = get_sentiment, indie_from_parsing_hidden

    def forward(self, *args, **kw_args):
        # TODO: data efficiency for mt
        base_returns = super().forward(*args, **kw_args)
        get_sentiment, indie_from_parsing_hidden = self._get_sentiment
        if indie_from_parsing_hidden:
            sentiment = get_sentiment(base_returns[5])
        else:
            sentiment = get_sentiment(base_returns[6])
        return base_returns + (sentiment,)