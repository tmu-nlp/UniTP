do_nothing = lambda x: x

def make_inf_gen(elem):
    while True:
        yield elem

inf_none_gen = make_inf_gen(None)
inf_zero_gen = make_inf_gen(0)

def cumu_slice(segments, interval = 0):
    start = 0
    for length in segments:
        end = start + length + interval
        yield start, end
        start = end