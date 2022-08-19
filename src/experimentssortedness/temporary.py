"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""
import numpy as np
from lange import ap
from numpy import nan
from scipy.stats import spearmanr

from sortedness.rank import (
    rank_by_distances,
    euclidean__n_vs_1,
)


# noinspection PyTypeChecker
def sortedness(X, X_, f=spearmanr, return_pvalues=False, weigher=None, normalized=True):
    result, pvalues = [], []
    if f is None:
        if weigher is None:
            weigher = lambda r: 1 / (1 + r)
        weights = [weigher(i) for i in range(len(X))]
        if normalized:
            lst = ap[1, 2, ..., len(X)].l
            woa = np.array(lst, dtype=np.float).reshape(len(lst), 1)
            wob = np.array(list(reversed(lst)), dtype=np.float).reshape(len(lst), 1)
            worst = ff(woa, wob, weights)
        for a, b in zip(X, X_):
            ra, rb = rank_by_distances(X, a), rank_by_distances(X_, b)
            t = ff(ra, rb, weights)
            if normalized:
                if t != 0:
                    t /= worst
                t = 1 - (2 * t)
            result.append(t)
            pvalues.append(nan)
    else:
        if weigher is not None:
            raise Exception("Cannot provide both 'f' and 'weigher'.")
        for a, b in zip(X, X_):
            corr, pvalue = f(euclidean__n_vs_1(X, a), euclidean__n_vs_1(X_, b))
            result.append(round(corr, 12))
            pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return result, pvalues
        # return list(zip(result, pvalues))
    return result


def ff(ranksa, ranksb, weights):
    t = 0
    for idxa, idxb in zip(ranksa, ranksb):
        mn, mx = sorted([int(idxa), int(idxb)])
        t += sum(weights[p] for p in range(mn, mx) if p >= 0)  # ignores point as a neighbor-of-itself
    return round(t, 12)
