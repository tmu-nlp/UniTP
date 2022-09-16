def isqrt(n):
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x

frac_eq = lambda a, b, n, o = 0: int(a * n) == int(b * n + o)
frac_neq = lambda a, b, n, o = 0: not frac_eq(a, b, n, o)

def is_bin_times(n):
    return 0 == (n & (n-1))

def harmony(*fractions):
    return len(fractions) / sum(1/f for f in fractions)

from math import sqrt, floor, log

def log_to_frac(*x_low_high):
    x, low, high = (log(x) for x in x_low_high)
    return (x - low) / (high - low)

def t_index(sid, b = 1):
    c = (2 - b) / (2 * b)
    lid = floor(sqrt(sid * 2 + c*c) - c)
    return lid, sid - s_index(lid, 0, b)

def s_index(lid, offset = 0, b = 1):
    return ((lid * (b * lid + (2 - b))) >> 1) + offset

def f_score(t1, t2, beta = 1):
    b = beta ** 2
    d = b * t1 + t2
    if d:
        return (1 + b) * (t1 * t2) / d
    return 0

def bit_fanout(bits):
    prev, result = bits, 0
    while bits:
        bits &= bits - 1
        if ((prev - bits) << 1) & prev == 0:
            result += 1
        prev = bits
    return result
# '1000 100 - 1' causes the lowest part changes '1000 011'
# bits &= bits - 1 examines the part and perserving higher parts '1000 000'
# (prev - bits) removes higher parts
# '1111 100 - 1 == 1111 001', 1111 000
# ((prev - bits) << 1) & prev, (100 - 011 << 1) & 100X 100 examines X
n_bit = lambda bits, base = 2: floor(log(bits, base)) + 1

def low_bit(bits):
    low = bits - 1; low &= bits
    return bits - low

def bit_span(bits):
    high = (1 << n_bit(bits)) - 1
    return high ^ (low_bit(bits) - 1)

from itertools import count
def location_gen(bits):
    for i in count():
        bit = 1 << i
        if bit > bits:
            break
        if bit & bits:
            yield i

def inv_sigmoid(y):
    return - log(1 / y - 1)

def uneven_split(threshold, score):
    score -= threshold
    if score > 0:
        score /= 1 - threshold
    else:
        score /= threshold
    return score

def lr_gen(list_or_tuple, start):
    n = len(list_or_tuple) - 1
    radius = 1
    while True:
        lhs = start - radius
        has_lhs = 0 <= lhs
        if has_lhs:
            yield list_or_tuple[lhs]
        rhs = start + radius
        has_rhs = rhs <= n
        if has_rhs:
            yield list_or_tuple[rhs]
        if has_lhs or has_rhs:
            radius += 1
            continue
        break

def itp(lhs, rhs, res):
    lhs_itp = (res - rhs)
    lhs_itp /= (lhs - rhs)
    rhs_itp = 1 - lhs_itp
    return lhs_itp, rhs_itp

if __name__ == '__main__':
    def max_nil_relay(n):
        if n < 3:
            return (0, 0)
        elif n == 3:
            return (0, 1)
        l = n >> 1
        r = n - l
        c = (l - 1) * (r - 1)
        t = (l - 1) + (r - 1)
        l, lt = max_nil_relay(l)
        r, rt = max_nil_relay(r)
        return (l + c + r, lt + t + rt)

    print('bottom', 'num_node', 'max_nil_ratio', 'relay_ratio', 'key_ratio')
    for i in tuple(range(2, 10)) + (16, 32, 64, 128):
        num_nil, num_relay = max_nil_relay(i)
        num_node = (i * (i - 1)) >> 1
        print('%6d %8d        %.4f      %.4f    %.4f' % (i, num_node, num_nil / num_node, num_relay / num_node, (i-1) / num_node))