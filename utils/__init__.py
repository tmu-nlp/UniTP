do_nothing = lambda x: x

def cumu_slice(segments, interval = 0):
    start = 0
    for length in segments:
        end = start + length + interval
        yield start, end
        start = end