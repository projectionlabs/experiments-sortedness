def none():
    """
    Some experiments with wtau:

    >>> l = list(range(17))
    >>> ll = [[i] for i in l]


    >>> # rank=l
    >>> f = lambda a,b: round(weightedtau(a, b, rank=l, weigher=lambda x:1/(1+x))[0], 12)

    >>> # -1.0              Total reversion.
    >>> a,b = np.array(ll), np.array(list(reversed(ll)))
    >>> b.ravel()
    array([16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
    >>> f(a, b)
    -1.0

    >>> # 0.000856397632     Random.
    >>> rnd = np.random.default_rng(4)
    >>> lst = ll[-1:] + ll[1:-1] + ll[0:1]
    >>> rnd.shuffle(lst)
    >>> b = np.array(lst)
    >>> b.ravel()
    array([ 8, 16,  1,  0, 15,  2,  9,  7, 10, 13,  6,  4,  3, 12,  5, 11, 14])
    >>> f(a, b)
    0.000856397632

    >>> # 0.21128423268     Swap first and last.
    >>> b = np.array(ll[-1:] + ll[1:-1] + ll[0:1])
    >>> b.ravel()
    array([16,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0])
    >>> f(a, b)
    0.21128423268

    >>> # 0.329870949736    Bring last to the front and shift others to the right.
    >>> b = np.roll(a, 1)
    >>> b.ravel()
    array([16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    >>> f(a, b)
    0.329870949736

    >>> # 0.558670392162    Swap second and last.
    >>> b = np.array(ll[0:1] + ll[-1:] + ll[2:-1] + ll[1:2])
    >>> b.ravel()
    array([ 0, 16,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  1])
    >>> f(a, b)
    0.558670392162

    >>> # 0.688004931539    Swap third and last.
    >>> b = np.array(ll[0:2] + ll[-1:] + ll[3:-1] + ll[2:3])
    >>> b.ravel()
    array([ 0,  1, 16,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  2])
    >>> f(a, b)
    0.688004931539

    >>> # 0.842933585279    Bring first to the end and shift others to the left.
    >>> b = np.roll(a, -1)
    >>> b.ravel()
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0])
    >>> f(a, b)
    0.842933585279

    >>> # 0.945487094974    Swap first two.
    >>> b = np.array(ll[1:2] + ll[0:1] + ll[2:])
    >>> b.ravel()
    array([ 1,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    >>> f(a, b)
    0.945487094974

    >>> # 0.995590867976    Swap last two.
    >>> b = np.array(ll[0:-2] + ll[-1:] + ll[-2:-1])
    >>> b.ravel()
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 15])
    >>> f(a, b)
    0.995590867976







    >>> # Multiplicative. Failed at 'Swap first and last'.
    >>> f = lambda a,b: round(weightedtau(a, b, rank=l, weigher=lambda x:1/(1+x), additive=False)[0], 12)

    >>> # -1.0              Total reversion.
    >>> a,b = np.array(ll), np.array(list(reversed(ll)))
    >>> b.ravel()
    array([16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
    >>> f(a, b)
    -1.0

    >>> # -0.053210202596     Random.
    >>> rnd = np.random.default_rng(4)
    >>> lst = ll[-1:] + ll[1:-1] + ll[0:1]
    >>> rnd.shuffle(lst)
    >>> b = np.array(lst)
    >>> b.ravel()
    array([ 8, 16,  1,  0, 15,  2,  9,  7, 10, 13,  6,  4,  3, 12,  5, 11, 14])
    >>> f(a, b)
    -0.053210202596

    >>> # -0.007387377741     Swap first and last.
    >>> b = np.array(ll[-1:] + ll[1:-1] + ll[0:1])
    >>> b.ravel()
    array([16,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0])
    >>> f(a, b)
    -0.007387377741

    >>> # 0.047302373749    Bring last to the front and shift others to the right.
    >>> b = np.roll(a, 1)
    >>> b.ravel()
    array([16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    >>> f(a, b)
    0.047302373749

    >>> # 0.578077727209    Swap second and last.
    >>> b = np.array(ll[0:1] + ll[-1:] + ll[2:-1] + ll[1:2])
    >>> b.ravel()
    array([ 0, 16,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  1])
    >>> f(a, b)
    0.578077727209

    >>> # 0.755365767039    Swap third and last.
    >>> b = np.array(ll[0:2] + ll[-1:] + ll[3:-1] + ll[2:3])
    >>> b.ravel()
    array([ 0,  1, 16,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  2])
    >>> f(a, b)
    0.755365767039

    >>> # 0.922338397599    Bring first to the end and shift others to the left.
    >>> b = np.roll(a, -1)
    >>> b.ravel()
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0])
    >>> f(a, b)
    0.922338397599

    >>> # 0.804739267261    Swap first two.
    >>> b = np.array(ll[1:2] + ll[0:1] + ll[2:])
    >>> b.ravel()
    array([ 1,  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    >>> f(a, b)
    0.804739267261

    >>> # 0.998564259318    Swap last two.
    >>> b = np.array(ll[0:-2] + ll[-1:] + ll[-2:-1])
    >>> b.ravel()
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 15])
    >>> f(a, b)
    0.998564259318






    >>> # rank=None (scores are reversed ranking)
    >>> f = lambda a,b: round(weightedtau(a, b, rank=None, weigher=lambda x:1/(1+x))[0], 12)

    >>> # -1.0              Total reversion.
    >>> ll = list(reversed(ll))
    >>> a, b = np.array(ll), np.array(list(reversed(ll)))
    >>> b.ravel()
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    >>> f(a, b)
    -1.0

    >>> # 0.000856397632     Random.
    >>> rnd = np.random.default_rng(4)
    >>> lst = ll[-1:] + ll[1:-1] + ll[0:1]
    >>> rnd.shuffle(lst)
    >>> b = np.array(lst)
    >>> b.ravel()
    array([ 8,  0, 15, 16,  1, 14,  7,  9,  6,  3, 10, 12, 13,  4, 11,  5,  2])
    >>> f(a, b)
    0.000856397632

    >>> # 0.21128423268     Swap first and last.
    >>> b = np.array(ll[-1:] + ll[1:-1] + ll[0:1])
    >>> b.ravel()
    array([ 0, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 16])
    >>> f(a, b)
    0.21128423268

    >>> # 0.329870949736    Bring last to the front and shift others to the right.
    >>> b = np.roll(a, 1)
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> f(a, b)
    0.329870949736

    >>> # 0.558670392162    Swap second and last.
    >>> b = np.array(ll[0:1] + ll[-1:] + ll[2:-1] + ll[1:2])
    >>> b.ravel()
    array([16,  0, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 15])
    >>> f(a, b)
    0.558670392162

    >>> # 0.688004931539    Swap third and last.
    >>> b = np.array(ll[0:2] + ll[-1:] + ll[3:-1] + ll[2:3])
    >>> b.ravel()
    array([16, 15,  0, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 14])
    >>> f(a, b)
    0.688004931539

    >>> # 0.842933585279    Bring first to the end and shift others to the left.
    >>> b = np.roll(a, -1)
    >>> b.ravel()
    array([15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 16])
    >>> f(a, b)
    0.842933585279

    >>> # 0.945487094974    Swap first two.
    >>> b = np.array(ll[1:2] + ll[0:1] + ll[2:])
    >>> b.ravel()
    array([15, 16, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
    >>> f(a, b)
    0.945487094974

    >>> # 0.995590867976    Swap last two.
    >>> b = np.array(ll[0:-2] + ll[-1:] + ll[-2:-1])
    >>> b.ravel()
    array([16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  0,  1])
    >>> f(a, b)
    0.995590867976







    >>> # rank=True (scores are reversed ranking)
    >>> f = lambda a,b: round(weightedtau(a, b, rank=True, weigher=lambda x:1/(1+x))[0], 12)

    >>> # -1.0              Total reversion.
    >>> a, b = np.array(ll), np.array(list(reversed(ll)))
    >>> b.ravel()
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    >>> f(a, b)
    -1.0

    >>> # 0.03793522443     Random.
    >>> rnd = np.random.default_rng(3)
    >>> lst = ll[-1:] + ll[1:-1] + ll[0:1]
    >>> rnd.shuffle(lst)
    >>> b = np.array(lst)
    >>> b.ravel()
    array([ 6,  3, 13, 16,  5,  8,  4,  1, 15, 14,  2,  0,  7, 12,  9, 10, 11])
    >>> f(a, b)
    0.03793522443

    >>> # 0.21128423268     Swap first and last.
    >>> b = np.array(ll[-1:] + ll[1:-1] + ll[0:1])
    >>> b.ravel()
    array([ 0, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 16])
    >>> f(a, b)
    0.21128423268

    >>> # 0.558670392162    Swap second and last.
    >>> b = np.array(ll[0:1] + ll[-1:] + ll[2:-1] + ll[1:2])
    >>> b.ravel()
    array([16,  0, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 15])
    >>> f(a, b)
    0.558670392162

    >>> # 0.586402267507    Bring last to the front and shift others to the right.
    >>> b = np.roll(a, 1)
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> f(a, b)
    0.586402267507

    >>> # 0.586402267507    Bring first to the end and shift others to the left.
    >>> b = np.roll(a, -1)
    >>> b.ravel()
    array([15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 16])
    >>> f(a, b)
    0.586402267507

    >>> # 0.688004931539    Swap third and last.
    >>> b = np.array(ll[0:2] + ll[-1:] + ll[3:-1] + ll[2:3])
    >>> b.ravel()
    array([16, 15,  0, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 14])
    >>> f(a, b)
    0.688004931539

    >>> # 0.945487094974    Swap first two.
    >>> b = np.array(ll[1:2] + ll[0:1] + ll[2:])
    >>> b.ravel()
    array([15, 16, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
    >>> f(a, b)
    0.945487094974

    >>> # 0.995590867976    Swap last two.
    >>> b = np.array(ll[0:-2] + ll[-1:] + ll[-2:-1])
    >>> b.ravel()
    array([16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  0,  1])
    >>> f(a, b)
    0.995590867976
    """
    pass
