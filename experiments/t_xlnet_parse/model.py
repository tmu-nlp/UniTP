from models.nccp import BaseRnnTree, penn_tree_config, torch, nn, Tensor
from utils.types import word_dim, num_ctx_layer, frac_2, frac_4, BaseType

E_SUB = S_LFT, S_AVG, S_SGT = 'leftmost average selfgate'.split()
subword_proc = BaseType(0, as_index = True, default_set = E_SUB)

from models.backend import contextual_type, activation_type
xlnt_leaves_config = dict(contextual   = contextual_type,
                          num_layers   = num_ctx_layer,
                          drop_out     = frac_4,
                          rnn_drop_out = frac_2,
                          model_dim    = word_dim,
                          activation   = activation_type,
                          subword_proc = subword_proc)
xlnet_penn_tree_config = penn_tree_config.copy()
xlnet_penn_tree_config['embed_layer'] = xlnt_leaves_config

from models.utils import squeeze_left
class XLNetLeaves(nn.Module):
    def __init__(self,
                 contextual,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 model_dim,
                 activation,
                 paddings,
                 subword_proc):
        super().__init__()
        from transformers import XLNetModel
        self._xlnet_model = XLNetModel.from_pretrained(model_key_name)
        self._xlnet_dp = nn.Dropout(drop_out)
        self._activation = activation()
        
        if num_layers == 0:
            self._is_linear = True
            self._to_word_emb = nn.Linear(768, model_dim + (subword_proc == S_SGT))
        else:
            self._is_linear = False
            self._to_word_emb = contextual(768, model_dim // 2 + (subword_proc == S_SGT), 
                                           num_layers,
                                           batch_first = True,
                                           bidirectional = True,
                                           dropout = rnn_drop_out if num_layers > 1 else 0)
            if subword_proc == S_SGT:
                self._rnn_word_gate = nn.Linear(2, 1)
        if subword_proc == S_SGT:
            self._gate_act = nn.Sigmoid()

        if paddings:
            self._bos = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad = True)
            self._eos = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad = True)
        self._paddings = paddings
        self._word_dim = model_dim
        self._subword_proc = subword_proc
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
            xl_hidden = self._xlnet_dp(xl_hidden)
            # xl_hidden = xl_hidden[:, :-2] # Bad idea: git rid of some [cls][sep]

        def transform_dim(xl_hidden):
            word_hidden = self._to_word_emb(xl_hidden)
            if self._is_linear:
                if self._subword_proc == S_SGT:
                    word_gate = self._gate_act(word_hidden[:, :, 0, None])
                    word_hidden = self._activation(word_hidden[:, :, 1:])
                    word_hidden = word_gate * word_hidden
                else:
                    word_hidden = self._activation(word_hidden)
            else:
                word_hidden = word_hidden[0]
                if self._subword_proc == S_SGT:
                    lw_gate = word_hidden[:, :,  0, None]
                    rw_gate = word_hidden[:, :, -1, None]
                    word_gate = torch.cat([lw_gate, rw_gate], dim = 2)
                    word_gate = self._rnn_word_gate(word_gate)
                    word_gate = self._gate_act(word_gate)
                    word_hidden = word_gate * word_hidden[:, :, 1:-1]
            return word_hidden

        if self._subword_proc == S_LFT:
            xl_hidden, _ = squeeze_left(xl_hidden, xl_start, as_existence = True, **squeeze_params)
            xl_base = transform_dim(xl_hidden) # use left most sub-word to save precious time!
        else:
            word_hidden = transform_dim(xl_hidden)
            xl_base, xl_cumu = squeeze_left(word_hidden, xl_start, as_existence = False, **squeeze_params)
            if self._subword_proc == S_AVG:
                xl_cumu[xl_cumu < 1] = 1 # prevent 0
                xl_base = xl_base / xl_cumu
        
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

class XLNetPennTree(nn.Module):
    def __init__(self,
                 paddings,
                 embed_layer,
                 **base_config):
        super().__init__()

        self._emb_layer = emb_layer = XLNetLeaves(paddings = paddings, **embed_layer)
        self._base_model = BaseRnnTree(emb_layer.embedding_dim, **base_config)
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