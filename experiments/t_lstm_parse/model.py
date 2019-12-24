from models.backend import contextual_type
from models.penn import BasePennTree, penn_tree_config, torch, nn, Tensor
from models.utils import PCA
from utils.types import true_type, false_type, word_dim, num_ctx_layer, frac_4, frac_2

leaves_config = dict(use_fasttext = true_type,
                     word_dim     = word_dim,
                     trainable    = false_type,
                     contextual   = contextual_type,
                     num_layers   = num_ctx_layer,
                     drop_out     = frac_4,
                     rnn_drop_out = frac_2)

class LstmLeaves(nn.Module):
    def __init__(self,
                 num_words,
                 initial_weight,
                 use_fasttext,
                 word_dim,
                 trainable,
                 contextual,
                 num_layers,
                 drop_out,
                 rnn_drop_out):
        super().__init__()

        # word
        st_dy_bound = 0
        st_emb_layer = dy_emb_layer = None
        if use_fasttext:
            num_special_tokens = num_words - initial_weight.shape[0]
            assert num_special_tokens >= 0
            if num_special_tokens > 0:
                if trainable:
                    initial_weight = torch.cat([torch.tensor(initial_weight), torch.rand(num_special_tokens, word_dim)], 0)
                    st_emb_layer = None
                    dy_emb_layer = nn.Embedding.from_pretrained(initial_weight)
                else:
                    assert word_dim == initial_weight.shape[1]
                    st_dy_bound = initial_weight.shape[0]
                    st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
                    dy_emb_layer = nn.Embedding(num_special_tokens, word_dim)
            elif trainable:
                dy_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
            else:
                st_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = True)
        else:
            assert trainable
            dy_emb_layer = nn.Embedding(num_words, word_dim)

        self._word_dim = word_dim
        self._st_dy_bound = st_dy_bound
        self._st_emb_layer = st_emb_layer
        self._dy_emb_layer = dy_emb_layer
        self._dp_layer = nn.Dropout(drop_out)
        self._pca_base = None

        # contextual
        if num_layers:
            if contextual in (nn.LSTM, nn.GRU):
                assert word_dim % 2 == 0
                self._contextual = contextual(word_dim,
                                              word_dim // 2,
                                              num_layers,
                                              bidirectional = True,
                                              batch_first = True,
                                              dropout = rnn_drop_out if num_layers > 1 else 0)
            else:
                raise NotImplementedError()
        else:
            self._contextual = None

    @property
    def embedding_dim(self):
        return self._word_dim

    def pca(self, word_emb, flush = False):
        # TODO: setup_pca with external
        if self._st_emb_layer is not None and self._pca_base is None:
            self._pca_base = PCA(self._st_emb_layer.weight)
        return self._pca_base(word_emb)

    def forward(self, word_idx):
        if self._st_dy_bound > 0:
            b_ = self._st_dy_bound
            c_ = word_idx < b_
            st_idx = torch.where(c_, word_idx, torch.zeros_like(word_idx))
            dy_idx = torch.where(c_, torch.zeros_like(word_idx), word_idx - b_)
            st_emb = self._st_emb_layer(st_idx)
            dy_emb = self._dy_emb_layer(dy_idx)
            static_emb = torch.where(c_.unsqueeze(-1), st_emb, dy_emb)
            bottom_existence = torch.ones_like(word_idx, dtype = torch.bool)
        else:
            emb_layer = self._st_emb_layer or self._dy_emb_layer
            static_emb = emb_layer(word_idx)
            bottom_existence = word_idx > 0

        static_emb = self._dp_layer(static_emb)

        if self._contextual is None:
            dynamic_emb = None
        else:
            dynamic_emb, _ = self._contextual(static_emb)
            dynamic_emb = dynamic_emb + static_emb # += does bad to gpu
            # dynamic_emb = self._dp_layer(dynamic_emb)

        return static_emb, dynamic_emb, bottom_existence


lstm_penn_tree_config = penn_tree_config.copy()
lstm_penn_tree_config['embed_layer'] = leaves_config

class LstmPennTree(nn.Module):
    def __init__(self,
                 num_words,
                 initial_weights,
                 embed_layer,
                 paddings,
                 **base_config):
        super().__init__()

        self._emb_layer = emb_layer = LstmLeaves(num_words, initial_weights, **embed_layer)
        self._base_model = BasePennTree(emb_layer.embedding_dim, **base_config)
        self._paddings = paddings

    def forward(self, word_idx, **kw_args):
        batch_size, batch_len = word_idx.shape
        static, dynamic, bottom_existence = self._emb_layer(word_idx)

        if dynamic is None:
            base_inputs = static
        else:
            base_inputs = dynamic

        base_returns = self._base_model(base_inputs, bottom_existence, **kw_args)

        return (batch_size, batch_len, static, dynamic) + base_returns