from models.penn import BasePennTree, penn_tree_config, torch, nn, Tensor

class XLNetPennTree(nn.Module):
    def __init__(self,
                 paddings,
                 embed_layer,
                 **base_config):
        super().__init__()

        self._emb_layer = emb_layer = XLNetLeaves(paddings = paddings, **embed_layer)
        self._base_model = BasePennTree(emb_layer.embedding_dim, **base_config)
        self._paddings = paddings

    def forward(self,
                word_idx,
                offset, xl_ids, xl_start, **kw_args):
        batch_size, batch_len = word_idx.shape

        static, dynamic = self._emb_layer(word_idx, offset, xl_ids, xl_start)

        if self._paddings:
            bottom_existence = torch.ones(batch_size, batch_len,
                                          dtype = torch.bool,
                                          device = word_idx.device)
        else:
            bottom_existence = word_idx > 0
        base_returns = self._base_model(dynamic, bottom_existence, **kw_args)

        return (batch_size, batch_len, static, dynamic) + base_returns

model_key_name = 'xlnet-base-cased'
_penn_to_xlnet = {'``': '"', "''": '"'}
from tqdm import tqdm
class XLNetDatasetHelper(object):
    tokenizer = None

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls.tokenizer is None:
            from transformers import XLNetTokenizer #, SPIECE_UNDERLINE
            cls.tokenizer = XLNetTokenizer.from_pretrained(model_key_name)
        return object.__new__(cls)

    def __init__(self, text, device):
        self._xl_ids   = []
        self._xl_start = []
        self._indices  = []
        self._max_len  = 0
        self._device = device
        if text is not None:
            for penn_words in tqdm(text, desc = 'XLNetTokenizer'):
                self.append(penn_words)
        
    def append(self, penn_words, check = False):
        text = None
        for pw in penn_words:
            pw = _penn_to_xlnet.get(pw, pw)
            if text is None:
                text = pw
            elif pw in '.,();':
                text += pw
            else:
                text += ' ' + pw
        xlnt_words = self.tokenizer.tokenize(text)
        xlnt_starts = []
        xlnt_offset = 0
        for i_, pw in enumerate(penn_words):
            xlnt_starts.append(True)
            xlnt_word = xlnt_words[i_ + xlnt_offset]
            if xlnt_word[0] == '▁':
                xlnt_word = xlnt_word[1:]
            if pw == xlnt_word:
                continue
            while xlnt_word != pw:
                xlnt_offset += 1
                try:
                    piece = xlnt_words[i_ + xlnt_offset]
                except:
                    import pdb; pdb.set_trace()
                if piece == '"': # -`` in ptb
                    piece = '``' if '``' in pw else "''"
                xlnt_word += piece
                xlnt_starts.append(False)
        word_idx = self.tokenizer.encode(text)
        if check:
            assert len(xlnt_words) == len(xlnt_starts), text + f" {' '.join(xlnt_words)}"
            if len(word_idx) - 2 != len(xlnt_words):
                import pdb; pdb.set_trace()
            if len(penn_words) != sum(xlnt_starts):
                import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        self._xl_ids.append(word_idx)
        self._xl_start.append(xlnt_starts)

    def buffer(self, idx):
        self._indices.append(idx)
        wlen = len(self._xl_ids[idx])
        if wlen > self._max_len:
            self._max_len = wlen

    def get(self):
        xl_ids, xl_start = [], []
        for idx in self._indices:
            wi = self._xl_ids  [idx]
            ws = self._xl_start[idx]
            len_diff = self._max_len - len(wi)
            xl_ids  .append(wi + [self.tokenizer.pad_token_id] * len_diff)
            xl_start.append(ws + [True] + [False] * (len_diff + 1))

        self._indices = []
        self._max_len = 0

        xl_ids = torch.tensor(xl_ids, device = self._device)
        xl_start = torch.tensor(xl_start, device = self._device)
        return dict(xl_ids = xl_ids, xl_start = xl_start)

from utils.types import word_dim, num_pre_layer, false_type
from models.backend import activation_type, contextual_type
xlnt_leaves_config = dict(contextual   = contextual_type,
                          num_layers   = num_pre_layer,
                          word_dim     = word_dim,
                          activation   = activation_type,
                          avg_subwords = false_type)
xlnet_penn_tree_config = penn_tree_config.copy()
xlnet_penn_tree_config['embed_layer'] = xlnt_leaves_config

from models.utils import squeeze_left
class XLNetLeaves(nn.Module):
    def __init__(self,
                 contextual,
                 num_layers,
                 word_dim,
                 activation,
                 paddings,
                 avg_subwords):
        super().__init__()
        from transformers import XLNetModel
        self._xlnet_model = XLNetModel.from_pretrained(model_key_name)
        if num_layers == 0:
            self._is_linear = True
            self._to_word_emb = nn.Linear(768, word_dim)
        else:
            self._is_linear = False
            self._to_word_emb = contextual(768, word_dim // 2, num_layers, batch_first = True, bidirectional = True)
        self._activation = activation()
        if paddings:
            self._bos = nn.Parameter(torch.randn(1, 1, word_dim), requires_grad = True)
            self._eos = nn.Parameter(torch.randn(1, 1, word_dim), requires_grad = True)
        self._paddings = paddings
        self._word_dim = word_dim
        self._avg_subwords = avg_subwords
        # self._xlnet_model.train()

    @property
    def embedding_dim(self):
        return self._word_dim

    def forward(self, word_idx, offset, xl_ids, xl_start):
        squeeze_params = dict(offset  = offset,
                              out_len = word_idx.shape[1],
                              get_rid_of_last_k = 1)

        with torch.no_grad():
            xl_hidden = self._xlnet_model(xl_ids)[0]
            # xl_hidden = xl_hidden[:, :-2] # Bad idea: git rid of some [cls][sep]

        def transform_dim(xl_hidden):
            word_hidden = self._to_word_emb(xl_hidden)
            if self._is_linear:
                word_hidden = self._activation(word_hidden)
            else:
                word_hidden = word_hidden[0]
            return word_hidden

        if self._avg_subwords:
            word_hidden = transform_dim(xl_hidden)
            xl_base, xl_cumu = squeeze_left(word_hidden, xl_start, as_existence = False, **squeeze_params)
            xl_cumu[xl_cumu < 1] = 1 # prevent 0
            xl_base = xl_base / xl_cumu
        else:
            xl_hidden, _ = squeeze_left(xl_hidden, xl_start, as_existence = True, **squeeze_params)
            xl_base = transform_dim(xl_hidden) # use left most sub-word to save precious time!
        
        if self._paddings: # will overwrite [cls][sep]
            bos, eos = self._paddings['word']
            bos = (word_idx == bos)
            eos = (word_idx == eos)
            bos.unsqueeze_(dim = 2)
            eos.unsqueeze_(dim = 2)
            xl_base = torch.where(bos, self._bos.expand_as(xl_base), xl_base) # 不要让nn做太多离散的决定，人来做！
            xl_base = torch.where(eos, self._eos.expand_as(xl_base), xl_base)
        else:
            non_nil = (word_idx > 0)
            non_nil.unsqueeze_(dim = 2)
            xl_base = xl_base * non_nil # in-place fails at FloatTensor
        return None, xl_base # just dynamic