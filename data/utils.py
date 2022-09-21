from data import NIL, PAD, BOS, EOS, UNK, USUB

def pre_word_base(model_config):
    if use := model_config.get('use'):
        if use.get('char_rnn'):
            return CharTextHelper

from utils.param_ops import get_sole_key, change_key
parsing_pnames = 'num_tokens', 'weight_fn', 'num_tags', 'num_labels', 'paddings'
sentiment_pnames = 'num_tokens', 'weight_fn', 'num_polars', 'paddings'
parsing_sentiment_pnames = set(parsing_pnames) | set(sentiment_pnames)
def post_word_base(model_cls, model_config, data_config, readers, pnames = parsing_pnames):
    i2vs = {c: r.i2vs for c, r in readers.items()}
    if single_corpus := (len(data_config) == 1):
        single_corpus = get_sole_key(data_config)
        i2vs = i2vs[single_corpus]
    reader_types = len(set(r.__class__ for r in readers.values()))
    for pname in pnames:
        param = {}
        for c, r in readers.items():
            if reader_types == 1:
                param[c] = r.get_to_model(pname)
            elif r.has_for_model(pname):
                param[c] = r.get_to_model(pname)
        if reader_types > 1:
            assert param
            if single_corpus := (len(param) == 1):
                single_corpus = get_sole_key(param)
        model_config[pname] = param[single_corpus] if single_corpus else param
    return model_cls(**model_config), i2vs

class TextHelper:
    def __init__(self, cache):
        self._cache = cache
        self._buffer  = []
        self._max_len  = 0
        self._drop_cache = {}

    def buffer(self, idx):
        wi, ws = self._cache[idx]
        self._buffer.append((wi, ws))

        wlen = len(wi)
        if wlen > self._max_len:
            self._max_len = wlen
        return wi, ws

    def a_secrete_buffer(self, cache):
        for wi, ws in cache:
            self._buffer.append((wi, ws))

            wlen = len(wi)
            if wlen > self._max_len:
                self._max_len = wlen

    def gen_from_buffer(self):
        for wi, ws in self._buffer:
            yield wi, ws, self._max_len - len(wi)

        self._buffer = []
        self._max_len = 0

    def get(self):
        raise NotImplementedError('TextHelper.get')

import torch
from tqdm import tqdm
class CharTextHelper(TextHelper):
    def __init__(self, text, alphabet_fn):
        cache = []
        pad_idx = alphabet_fn(PAD)
        for words in tqdm(text, 'CharTextHelper'):
            char_seq = [pad_idx]
            segment = [0]
            for word in words:
                char_seq.extend(alphabet_fn(x) for x in word)
                char_seq.append(pad_idx)
                segment.append(len(word) + 1 + segment[-1])
            cache.append((char_seq, segment))
        super().__init__(cache)
        
    def get(self):
        char_idx = []
        for wi, ws, len_diff in self.gen_from_buffer():
            char_idx.append(wi + [0] * len_diff)
        char_idx = torch.tensor(char_idx, device = device)
        return dict(sub_idx = char_idx)

from os.path import join
from data.io import load_i2vs, get_fasttext
from data.vocab import VocabKeeper
class ParsingReader(VocabKeeper):
    def __init__(
            self,
            penn,
            unify_sub   = True,
            nil_as_pads = True):
        vocab_path = penn.local_path
        vocab_size = penn.vocab_size
        token_type = 'word' if penn.token is None else penn.token

        i2vs = load_i2vs(vocab_path, token_type, 'tag', 'label')
        change_key(i2vs, token_type, 'token')
        fields = ('token', 'tag', 'label')

        oovs = {}
        if unify_sub and 'label' in i2vs:
            labels = [t for t in i2vs['label'] if t[0] not in '#_']
            oovs['label'] = len(labels)
            labels.append(USUB)
            i2vs['label'] = labels

        paddings = {}
        if nil_as_pads:
            def token_fn(token_type):
                weights = get_fasttext(join(vocab_path, token_type + '.vec'))
                weights[0] = 0 # [nil ...]
                return weights

        else:
            def token_fn(token_type):
                return get_fasttext(join(vocab_path, token_type + '.vec'))[1:]
        
        # extra_fn = {}
        # if extra_text_helper is CharTextHelper:
        #     i2vs.update(vocab_path, 'char')
        #     assert token_type == 'word'
        #     extra_fn['char_weight_fn'] = lambda: token_fn('char')
        #     fields = field + ('char',)

        if vocab_size:
            assert token_type == 'word', 'No size limitation be onto chars'
            words = i2vs['token']
            assert vocab_size <= len(words)
            if nil_as_pads:
                unk_id = vocab_size - 1
                words = words[:unk_id] # for UNK
                oovs['token'] = unk_id
                words.append(UNK)
            else:
                unk_id = vocab_size - 3
                words = words[:unk_id] # for UNK, BOS, EOS
                oovs['token'] = unk_id
                words.append(UNK)
                for field in ('token', 'tag', 'label'): # NIL is a good label
                    i2v = i2vs[field]
                    assert i2v.pop(0) == NIL
                    i2v += [BOS, EOS]
                    num = len(i2v); paddings[field] = (num - 2, num - 1)

            token_vec_fn = lambda: token_fn(token_type)[:unk_id + 1]
        else:
            if not nil_as_pads:
                for field in ('token', 'tag', 'label'): # append pads
                    i2v = i2vs[field]
                    i2v += [BOS, EOS]
                    num = len(i2v); paddings[field] = (num - 2, num - 1)

            token_vec_fn = lambda: token_fn(token_type)
        super().__init__(fields, i2vs, oovs, paddings, weight_fn = token_vec_fn)#, **extra_fn)

    @property
    def unk_id(self):
        return self._oovs.get('token')