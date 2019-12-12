def isqrt(n):
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x

def is_bin_times(n):
    return 0 == (n & (n-1))

def harmony(fractions):
    return len(fractions) / sum(1/f for f in fractions)

from math import sqrt, floor
def t_index(sid, b = 1):
    c = (2 - b) / (2 * b)
    lid = floor(sqrt(sid * 2 + c*c) - c)
    return lid, sid - s_index(lid, 0, b)

def s_index(lid, offset = 0, b = 1):
    return ((lid * (b * lid + (2 - b))) >> 1) + offset

if __name__ == '__main__1':
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