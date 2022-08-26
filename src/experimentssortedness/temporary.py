"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""

import numpy as np
from numpy import argsort
from numpy.linalg import norm
from scipy.stats import weightedtau

from sortedness.rank import euclidean__n_vs_1


def sortedness(X, X_, f=weightedtau, return_pvalues=False, **kwargs):
    """
    >>> ll = [[i] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> r = sortedness(a, b)
    >>> min(r), max(r)
    (-1.0, 0.998638259786)
    """
    result, pvalues = [], []
    for a, b in zip(X, X_):
        distances_a = norm(X - a, axis=1)
        distances_b = norm(X_ - b, axis=1)
        indexes = argsort(distances_a, axis=0, kind="stable")
        # Scores for f=wtau go in the opposite direction of ranks.
        scores_a = -distances_a[indexes[1:]]
        scores_b = -distances_b[indexes[1:]]
        corr, pvalue = f(scores_a, scores_b, **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result
