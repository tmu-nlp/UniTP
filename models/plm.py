from utils.types import BaseType
E_SUB = S_LFT, S_RGT, S_AVG, S_SGT = 'leftmost rightmost average selfgate'.split()
subword_proc = BaseType(0, as_index = True, default_set = E_SUB)

from utils.types import num_ctx_layer_0, frac_0, true_type
from models.types import rnn_module_type, activation_type
plm_leaves_config = dict(contextual   = rnn_module_type,
                         num_layers   = num_ctx_layer_0,
                         drop_out     = frac_0,
                         rnn_drop_out = frac_0,
                         activation   = activation_type,
                         subword_proc = subword_proc,
                         sum_weighted_layers = true_type)

from models.utils import condense_helper, condense_left
from models.backend import torch, nn, init, math, BottomOutput
class PreLeaves(nn.Module):
    has_static_pca = False # class property

    def __init__(self,
                 model_key_name,
                 model_dim,
                 contextual,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 activation,
                 paddings,
                 subword_proc,
                 sum_weighted_layers,
                 **kwargs_forwarding):
        super().__init__(model_dim, **kwargs_forwarding) # paddings = paddings, 

        from transformers import AutoModel
        self._pre_model = AutoModel.from_pretrained(model_key_name)
        self._dp_layer = nn.Dropout(drop_out)
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
            bos = torch.empty(1, 1, model_dim)
            eos = torch.empty(1, 1, model_dim)
            self._bos = nn.Parameter(bos, requires_grad = True)
            self._eos = nn.Parameter(eos, requires_grad = True)
            bound = 1 / math.sqrt(model_dim)
            init.uniform_(self._bos, -bound, bound)
            init.uniform_(self._eos, -bound, bound)

        if sum_weighted_layers:
            n_layer = getattr(self._pre_model, 'n_layer', 12)
            layer_weights = torch.empty(n_layer + 1, 1, 1, 1) # TODO: better to use layers[n:]?
            self._layer_weights = nn.Parameter(layer_weights, requires_grad = True)
            self._softmax = nn.Softmax(dim = 0)
            bound = 1 / math.sqrt(n_layer)
            init.uniform_(self._layer_weights, -bound, bound)
            self._pre_model.config.__dict__['output_hidden_states'] = True
        else:
            self._layer_weights = None

        self._paddings = paddings
        self._word_dim = model_dim
        self._subword_proc = subword_proc
        # self._pre_model.train()

    @property
    def embedding_dim(self):
        return self._word_dim

    def forward(self, word_idx, tune_pre_trained,
                offset, plm_idx, plm_start, squeeze_existence,
                ignore_logits = False, 
               **kwargs):
        batch_size, batch_len = word_idx.shape # just provide shape and nil info

        if tune_pre_trained:
            plm_outputs = self._pre_model(plm_idx, output_hidden_states = True)
            # xl_hidden = xl_hidden[:, :-2] # Bad idea: git rid of some [cls][sep]
        else:
            with torch.no_grad():
                plm_outputs = self._pre_model(plm_idx, output_hidden_states = True)
        
        if self._layer_weights is None:
            xl_hidden = plm_outputs.last_hidden_state
        else:
            layer_weights = self._softmax(self._layer_weights) # [13, b, s, 768]
            xl_hidden = torch.stack(plm_outputs.hidden_states)
            xl_hidden = (xl_hidden * layer_weights).sum(dim = 0)

        xl_hidden = self._dp_layer(xl_hidden)

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

        if self._subword_proc in (S_LFT, S_RGT):
            if self._subword_proc == S_LFT: # i dis #agree with it
                xl_pointer = plm_start      # 1 1   0      1    1
            else:                           # 1 0   1      1    1
                xl_pointer = torch.cat([plm_start[:, 1:], torch.ones_like(plm_start[:, :1])], dim = 1)
            helper = condense_helper(xl_pointer, as_existence = True, offset = offset, get_rid_of_last_k = 1)
            xl_hidden = condense_left(xl_hidden, helper, out_len = batch_len)
            xl_base = transform_dim(xl_hidden) # use left most sub-word to save precious time!
        else:
            word_hidden = transform_dim(xl_hidden)
            helper = condense_helper(plm_start, as_existence = False, offset = offset, get_rid_of_last_k = 1)
            if self._subword_proc == S_AVG:
                xl_base, xl_cumu = condense_left(word_hidden, helper, out_len = batch_len, get_cumu = True)
                xl_cumu[xl_cumu < 1] = 1 # prevent 0
                xl_base = xl_base / xl_cumu
            else:
                xl_base = condense_left(word_hidden, helper, out_len = batch_len)
        
        if self._paddings: # will overwrite [cls][sep]
            bos, eos = self._paddings['word']
            bos = (word_idx == bos)
            eos = (word_idx == eos)
            bos.unsqueeze_(dim = 2)
            eos.unsqueeze_(dim = 2)
            xl_base = torch.where(bos, self._bos.expand_as(xl_base), xl_base) # 不要让nn做太多离散的决定，人来做！
            xl_base = torch.where(eos, self._eos.expand_as(xl_base), xl_base)
            non_nil = torch.ones(batch_size, batch_len, 1, dtype = torch.bool, device = word_idx.device)
        else:
            non_nil = (word_idx > 0)
            non_nil.unsqueeze_(dim = 2)
            xl_base = xl_base * non_nil # in-place fails at FloatTensor
        if squeeze_existence:
            non_nil = non_nil.squeeze(dim = 2)
        base_returns = super().forward(non_nil, xl_base, ignore_logits, **kwargs)
        return (BottomOutput(batch_size, batch_len, xl_base, None),) + base_returns

    @property
    def message(self):
        messages = []
        if self._layer_weights is not None:
            _layer_weights = self._layer_weights.detach().reshape(-1)
            _layer_weights = self._softmax(_layer_weights).cpu()
            min_val = _layer_weights.min()
            max_val = _layer_weights.max()
            _layer_weights = (_layer_weights - min_val) / (max_val - min_val)
            from utils.str_ops import height_ratio, str_ruler
            msg_0 = 'Layer weights: '
            msg_1 = ''.join(height_ratio(w) for w in _layer_weights.numpy())
            msg_2 = f'[{min_val:.2f}, {max_val:.2f}]'
            msg_3 = '\n               '
            msg_4 = str_ruler(len(msg_1))
            messages.append(msg_0 + msg_1 + msg_2 + msg_3 + msg_4)
        if hasattr(super(), 'message') and (message := super().message):
            messages.append(message)
        if messages:
            return '\n'.join(messages)


xlnet_model_key = 'xlnet-base-cased'
gbert_model_key = 'bert-base-german-cased'
class XLNetLeaves(PreLeaves):
    def __init__(self, input_layer, **kwargs):
        paddings = kwargs.get('paddings') # TODO remove this ugly piece...
        if isinstance(paddings, dict) and all(isinstance(p, dict) for p in paddings.values()):
            pd = None
            for p in paddings.values():
                if pd is None:
                    pd = p
                else:
                    assert pd == p
            kwargs['paddings'] = pd
        super().__init__(xlnet_model_key, **input_layer, **kwargs)

class GBertLeaves(PreLeaves):
    def __init__(self, input_layer, **kwargs):
        super().__init__(gbert_model_key, **input_layer, **kwargs)


_penn_to_xlnet = {'``': '"', "''": '"'}
from unidecode import unidecode
from multiprocessing import Pool
from data.utils import TextHelper
class PreDatasetHelper(TextHelper):
    def __init__(self, text, *args):
        with Pool() as p:
            cache = p.map(self._append, text)
        # cache =  []
        # from tqdm import tqdm
        # for penn_words in tqdm(text, desc = self.tknz_name):
        #     cache.append(self._append(penn_words))
        super().__init__(cache)

    @classmethod
    def _append(cls, penn_words, check = True):
        text = cls._adapt_text_for_tokenizer(penn_words)
        xlnt_words  = cls.tokenizer.tokenize(text)
        word_idx    = cls.tokenizer.encode(text)
        # import pdb; pdb.set_trace()
        xlnt_starts = cls._start(penn_words, xlnt_words)
        if check:
            assert len(xlnt_words) == len(xlnt_starts), text + f" {' '.join(xlnt_words)}"
            if len(word_idx) - 2 != len(xlnt_words):
                import pdb; pdb.set_trace()
            if len(penn_words) != sum(xlnt_starts):
                import pdb; pdb.set_trace()
        return word_idx, xlnt_starts
        
    def get(self):
        from utils.types import device
        plm_idx, plm_start = [], []
        start, end, pad_token_id = self.start_end
        for wi, ws, len_diff in self.gen_from_buffer():
            plm_idx  .append(wi + [pad_token_id] * len_diff)
            plm_start.append(start + ws + [True] + [False] * (len_diff + end)) # TODO check!
        plm_idx   = torch.tensor(plm_idx,   device = device)
        plm_start = torch.tensor(plm_start, device = device)
        return dict(plm_idx = plm_idx, plm_start = plm_start)

    @staticmethod
    def _adapt_text_for_tokenizer(penn_words):
        raise NotImplementedError('PreDatasetHelper._adapt_text_for_tokenizer')

    @staticmethod
    def _start(penn_words, xlnt_words):
        raise NotImplementedError('PreDatasetHelper._start')
    

class XLNetDatasetHelper(PreDatasetHelper):
    tokenizer = None
    tknz_name = 'XLNetTokenizer'

    @classmethod
    def __new__(cls, *args, **kwargs):
        # sent <sep> <cls> <pad>
        # 1234   0     0     0   # 0 was truncated
        if cls.tokenizer is None:
            from transformers import AutoTokenizer
            cls.tokenizer = t = AutoTokenizer.from_pretrained(xlnet_model_key)
            cls.start_end = [], 1, t.pad_token_id
        return object.__new__(cls)

    # @classmethod
    # def for_discontinuous(cls, *args, **kwargs):
    #     # sent <sep> <cls> <pad>
    #     # 2345   0     0     0   # 0 was truncated, 1 is <0>
    #     if cls.tokenizer is None:
    #         from transformers import XLNetTokenizer #, SPIECE_UNDERLINE
    #         cls.tokenizer = t = AutoTokenizer.from_pretrained(xlnet_model_key)
    #         cls.start_end = [False], 0, t.pad_token_id
    #     return cls(*args, **kwargs)

    @staticmethod
    def _adapt_text_for_tokenizer(penn_words):
        text = None
        for pw in penn_words:
            pw = _penn_to_xlnet.get(pw, pw)
            if text is None:
                text = pw
            elif pw in '.,();':
                text += pw
            else:
                text += ' ' + pw
        return text

    @staticmethod
    def _start(penn_words, xlnt_words):
        xlnt_starts = []
        xlnt_offset = 0
        for i_, pw in enumerate(penn_words):
            xlnt_starts.append(True)
            xlnt_word = xlnt_words[i_ + xlnt_offset]
            if xlnt_word[0] == '▁':
                xlnt_word = xlnt_word[1:]
            if pw == xlnt_word:
                continue
            while xlnt_word != pw and xlnt_word != unidecode(pw):
                xlnt_offset += 1
                # clichés != cliches
                piece = xlnt_words[i_ + xlnt_offset]
                if piece == '"': # -`` in ptb
                    piece = '``' if '``' in pw else "''"
                xlnt_word += piece
                xlnt_starts.append(False)
                assert len(xlnt_word) <= len(pw), xlnt_word + '+' + pw
        return xlnt_starts


class GBertDatasetHelper(PreDatasetHelper):
    # <cls> sent <sep> <pad>
    #   0   2345   0     0
    tokenizer = None
    tknz_name = 'GermanBertTokenizer'

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls.tokenizer is None:
            from transformers import AutoTokenizer
            cls.tokenizer = t = AutoTokenizer.from_pretrained(gbert_model_key)
            cls.start_end = [False], 0, t.pad_token_id
        return object.__new__(cls)

    @classmethod
    def _adapt_text_for_tokenizer(cls, penn_words):
        text = None
        unk_token_id = cls.tokenizer.unk_token_id
        encode_func  = cls.tokenizer.encode
        for wid, pw in enumerate(penn_words):
            pw = _penn_to_xlnet.get(pw, pw)
            if unk_token_id in encode_func(pw)[1:-1]:
                # _pw = pw
                pw = unidecode(pw)
                penn_words[wid] = pw
                # print(_pw, pw)
            if text is None:
                text = pw
            elif pw in '.,();':
                text += pw
            else:
                text += ' ' + pw
        return text

    @classmethod
    def _start(cls, penn_words, xlnt_words):
        xlnt_starts = []
        xlnt_offset = 0
        unk_token = cls.tokenizer.unk_token
        for i_, pw in enumerate(penn_words):
            xlnt_starts.append(True)
            xlnt_word = xlnt_words[i_ + xlnt_offset]
            if pw == xlnt_word or xlnt_word == '"' and pw in ('``', "''"):
                continue
            elif xlnt_word == unk_token:
                # print('Unknown word:', pw)
                continue
            while xlnt_word != pw:
                xlnt_offset += 1
                try:
                    piece = xlnt_words[i_ + xlnt_offset]
                except:
                    import pdb; pdb.set_trace()
                if piece.startswith('##'):
                    piece = piece[2:]
                xlnt_word += piece
                xlnt_starts.append(False)
        return xlnt_starts