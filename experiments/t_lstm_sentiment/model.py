from experiments.t_lstm_parse.model import PennRnnTree, model_type
from models.backend import activation_type, nn, logit_type
from models.utils import get_logit_layer
from utils.types import true_type, hidden_dim, frac_4, BaseType, valid_size
from models.loss import get_decision, get_decision_with_value, get_loss

sentiment_layer = dict(indie_from_parsing_hidden = true_type,
                       indie_hidden_dim = hidden_dim,
                       indie_activation = activation_type,
                       indie_drop_out   = frac_4,
                       logit_type = logit_type)

model_type = model_type.copy()
model_type['sentiment_layer'] = sentiment_layer
model_type['tag_label_layer'] = model_type['tag_label_layer'].copy()
model_type['tag_label_layer']['hidden_dim'] = BaseType(None, as_exception = True, validator = valid_size)

class StanRnnTree(PennRnnTree):
    def __init__(self,
                 num_polars,
                 sentiment_layer,
                 *base_args,
                 **base_kwargs):
        super().__init__(*base_args, **base_kwargs)

        indie_from_parsing_hidden = sentiment_layer['indie_from_parsing_hidden']
        indie_hidden_dim = sentiment_layer['indie_hidden_dim']
        indie_activation = sentiment_layer['indie_activation']
        indie_drop_out   = sentiment_layer['indie_drop_out']
        Net, argmax, score_fn = get_logit_layer(sentiment_layer['logit_type'])
        
        if indie_from_parsing_hidden:
            if indie_hidden_dim:
                self._indie_hidden = a = nn.Linear(self.model_dim, indie_hidden_dim)
                self._activation = b =indie_activation()
                self._dp_layer = c = nn.Dropout(indie_drop_out)
                self._sentiment = d = Net(indie_hidden_dim, num_polars)
                final_layer = d
                def get_sentiment(x):
                    return d(c(b(a(x))))
            else:
                self._sentiment = get_sentiment = final_layer = Net(self.model_dim, num_polars)
        else:
            self._sentiment = get_sentiment = final_layer = Net(base_kwargs['tag_label_layer']['hiddin_dim'], num_polars)
        self._sentiment_argmax = argmax
        self._sentiment_finale = final_layer
        self._sentiment_score_fn = score_fn(dim = 2)
        self._get_sentiment = get_sentiment, indie_from_parsing_hidden

    def forward(self, *args, is_sentiment = False, **kw_args):
        base_returns = super().forward(*args, ingore_logits = is_sentiment, **kw_args)
        if is_sentiment:
            get_sentiment, indie_from_parsing_hidden = self._get_sentiment
            if indie_from_parsing_hidden:
                sentiment = get_sentiment(base_returns[5])
            else:
                sentiment = get_sentiment(base_returns[6])
            return base_returns + (sentiment,)
        return base_returns

    def get_polar_decision(self, logits):
        return get_decision(self._sentiment_argmax, logits)

    def get_polar_decision_with_value(self, logits):
        return get_decision_with_value(self._sentiment_score_fn, logits)

    def get_polar_loss(self, logits, batch, height_mask):
        return get_loss(self._sentiment_argmax, logits, batch, self._sentiment_finale, height_mask, 'polar')