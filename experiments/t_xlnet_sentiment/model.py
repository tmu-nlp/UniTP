from experiments.t_xlnet_parse.model import XLNetPennTree, model_type
from models.sentiment import SentimentExtention, inject_sentiment_type

model_type = inject_sentiment_type(model_type)

class StanXLNetTree(XLNetPennTree, SentimentExtention):
    def __init__(self,
                 num_polars,
                 sentiment_layer,
                 *base_args,
                 **base_kwargs):
        super().__init__(*base_args, **base_kwargs,
                         num_polars = num_polars,
                         _model_dim = base_kwargs['model_dim'],
                         hiddin_dim = base_kwargs['tag_label_layer']['hidden_dim'],
                         **sentiment_layer)

    def forward(self, *args, is_sentiment = False, **kw_args):
        base_returns = super().forward(*args, ingore_logits = is_sentiment, **kw_args)
        if is_sentiment:
            return self.extend(base_returns)
        return base_returns