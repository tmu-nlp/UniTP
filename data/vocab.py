from collections import namedtuple

class VocabKeeper:
    def __init__(self, fields, i2vs, oovs, paddings, **to_model):
        self._paddings = paddings
        to_model['paddings'] = paddings
        self._to_model = {}
        self._oovs = oovs
        self.update(fields, i2vs, **to_model)
        self._loaded_ds = {}

    def update(self, fields, i2vs, **to_model):
        sizes, i2vs, v2is = encapsulate_vocabs(i2vs, self._oovs)
        self._fields = f = namedtuple('Vocab', fields)
        self._i2vs = f(**i2vs)
        self._v2is = f(**v2is)
        self._sizes = f(**sizes)
        if to_model:
            if self._to_model:
                for k,v in self._to_model.items():
                    if k not in to_model:
                        to_model[k] = v # perserve old values
            self._to_model = to_model
        else:
            to_model = self._to_model
        to_model.update({'num_' + simple_plural(k) :v for k,v in sizes.items()})

    def update_to_model(self, **kw_args):
        self._to_model.update(kw_args)

    def change_oovs(self, field, offset):
        if field in self._oovs:
            self._oovs[field] += offset

    @property
    def sizes(self):
        return self._sizes

    @property
    def i2vs(self):
        return self._i2vs

    @property
    def v2is(self):
        return self._v2is

    @property
    def paddings(self):
        return self._paddings

    @property
    def loaded_ds(self):
        return self._loaded_ds

    def has_for_model(self, name):
        return name in self._to_model

    def get_to_model(self, name):
        return self._to_model[name]


def simple_plural(word):
    if word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    return word + 's'

def encapsulate_vocabs(i2vs, oovs):
    def inner(i2v, f): # python bug: namespace
        size = len(i2v)
        v2i = {v:i for i,v in enumerate(i2v)}
        
        if f in oovs: # replace the dict function
            oov = oovs[f]
            v2i_func = lambda v: v2i.get(v, oov)
            assert oov in range(size)
        else:
            v2i_func = v2i.get
        return size, v2i_func

    v2is = {}
    vlen = {}
    for f, i2v in i2vs.items():
        vl, v2i = inner(i2v, f)
        vlen[f] = vl
        v2is[f] = v2i
        i2vs[f] = tuple(i2v)

    return vlen, i2vs, v2is