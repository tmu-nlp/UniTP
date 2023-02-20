from numpy.random import choice

pairs = ('🔥.🎃', '⛄️.❄️', '⛈.☁️', '🌚.🌝', '🌞.🌜', '👻.😇', '🌕.🌗', '🍄.🌧')
pairs = pairs + ('💊.🦠', '🐚.🦕', '🎓.📖', '🍝.🥫', '🧨.💥', '💎.💥')
pairs = pairs + ('🍙.🍤', '👘.🧶', '🍱.🥟', '🎋.🎐', '🍜.🧂', '🍶.🍺')
pairs = pairs + ('🥥🌴🌱', '🥪🍞🌾', '🦆🐥🐣🥚', '🧮🤔🛹🦧', '🎹🎺🥁', '🌏🌞☢️', '🤖👽🚀')


def get_train_validation_pair():
    ngram = choice(pairs)
    if '.' in ngram:
        return ngram.split('.')
    a = tuple(a+b for a,b in zip(ngram, ngram[1:]))
    return choice(a)