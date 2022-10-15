from itertools import repeat, tee

import pathos.multiprocessing as mp
from numpy import lexsort, array, empty, intp, arange

from .wtau import _weightedrankedtau, _invert_in_place  # wtau.pyx


# TODO Confirm scipy LICENSE attribution for pyx
def parwtau(scoresX, scoresX_, npoints, R=True, parallel=True, **kwargs):
    """**kwargs is for ProcessingPool, e.g.: npcus=8"""
    npositions = len(scoresX)
    if len(scoresX.shape) == 2:
        perms_source = (lexsort((scoresX_[i], scoresX[i])) for i in range(npoints))
    else:
        lex = lexsort((scoresX_, scoresX))
        perms_source = (lex.copy() for i in range(npoints))
    perms, perms_for_r = tee(perms_source)
    add = R is True

    def genR(perms_gen, scores):
        for i in range(npoints):
            # print("s", i)
            p = next(perms_gen)
            if R is True or R is None:
                r = p.copy()
                # print(r)
                _invert_in_place(r)
            elif R is False:
                r = arange(npositions, dtype=intp)
            else:
                r = R[:, i]
            yield r

    exchanges_weights = (array([0.0]) for i in range(npoints))
    temps = (empty(npositions, dtype=int) for i in range(npoints))

    pmap = mp.ProcessingPool(**kwargs).imap if parallel else map
    if not add:
        jobs = repeat(scoresX), repeat(scoresX_), genR(perms_for_r, scoresX), repeat(npositions), perms, exchanges_weights, temps
        return array(list(pmap(_weightedrankedtau, *jobs)))

    def f(scoresX, scoresX_, rank, rank_, perm, perm_, exchanges_weight, temp):
        return (_weightedrankedtau(scoresX, scoresX_, rank, npositions, perm, exchanges_weight, temp) +
                _weightedrankedtau(scoresX_, scoresX, rank_, npositions, perm_, exchanges_weight, temp)) / 2

    scores = lambda M: (M[i] for i in range(npoints))
    if len(scoresX_.shape) == 2:
        perms_source_ = (lexsort((scoresX[i], scoresX_[i])).copy() for i in range(npoints))
    else:
        lex = lexsort((scoresX, scoresX_))
        perms_source_ = (lex.copy() for i in range(npoints))
    perms_, perms_for_r_ = tee(perms_source_)
    jobs = scores(scoresX), scores(scoresX_), genR(perms_for_r, scoresX), genR(perms_for_r_, scoresX_), perms, perms_, exchanges_weights, temps
    return array(list(pmap(f, *jobs)))


parwtau.isparwtau = True
