from models.loss import get_loss, sorted_decisions, sorted_decisions_with_values
from models.utils import get_logit_layer
from models.backend import nn

from utils.types import true_type, hidden_dim, frac_4, BaseType, valid_size
from models.types import activation_type, logit_type
special_hidden_dim = BaseType(None, as_exception = True, validator = valid_size)

def inject_sentiment_type(model_type):
    sentiment_layer = dict(indie_from_parsing_hidden = true_type,
                           indie_hidden_dim = hidden_dim,
                           indie_activation = activation_type,
                           indie_drop_out   = frac_4,
                           logit_type = logit_type)


    model_type = model_type.copy()
    model_type['sentiment_layer'] = sentiment_layer
    model_type['tag_label_layer'] = model_type['tag_label_layer'].copy()
    model_type['tag_label_layer']['hidden_dim'] = special_hidden_dim
    return model_type


class SentimentExtention(nn.Module):
    def __init__(self,
                 num_polars,
                 indie_from_parsing_hidden,
                 indie_hidden_dim,
                 indie_activation,
                 indie_drop_out,
                 logit_type,
                 _model_dim,
                 hiddin_dim):
        # indie_from_parsing_hidden = sentiment_layer['indie_from_parsing_hidden']
        # indie_hidden_dim = sentiment_layer['indie_hidden_dim']
        # indie_activation = sentiment_layer['indie_activation']
        # indie_drop_out   = sentiment_layer['indie_drop_out']
        super().__init__()
        Net, argmax, score_fn = get_logit_layer(logit_type) # sentiment_layer['logit_type'])
        
        if indie_from_parsing_hidden:
            if indie_hidden_dim:
                self._indie_hidden = a = nn.Linear(_model_dim, indie_hidden_dim)
                self._activation = b =indie_activation()
                self._dp_layer = c = nn.Dropout(indie_drop_out)
                self._sentiment = d = Net(indie_hidden_dim, num_polars)
                final_layer = d
                def get_sentiment(x):
                    return d(c(b(a(x))))
            else:
                self._sentiment = get_sentiment = final_layer = Net(_model_dim, num_polars)
        else:
            self._sentiment = get_sentiment = final_layer = Net(hiddin_dim, num_polars)
        self._sentiment_argmax = argmax
        self._sentiment_finale = final_layer
        self._sentiment_score_fn = score_fn(dim = 2)
        self._get_sentiment = get_sentiment, indie_from_parsing_hidden

    def extend(self, base_returns):
        get_sentiment, indie_from_parsing_hidden = self._get_sentiment
        if indie_from_parsing_hidden:
            sentiment = get_sentiment(base_returns[4])
        else:
            sentiment = get_sentiment(base_returns[5])
        return base_returns + (sentiment,)

    def get_polar_decisions(self, logits):
        return sorted_decisions(self._sentiment_argmax, 5, logits)

    def get_polar_decisions_with_values(self, logits):
        return sorted_decisions_with_values(self._sentiment_score_fn, 5, logits)

    def get_polar_loss(self, logits, top3_polar_logits, batch, height_mask, weight_mask):
        polar_loss = get_loss(self._sentiment_finale, self._sentiment_argmax, logits, batch, True, height_mask, weight_mask, 'polar')
        if top3_polar_logits is not None:
            polar_loss += get_loss(self._sentiment_finale, self._sentiment_argmax, top3_polar_logits, batch, None, 'top3_polar')
        return polar_loss