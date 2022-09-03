"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""

import numpy as np
from numpy import eye
from scipy.spatial.distance import cdist
from scipy.stats import weightedtau


def sortedness(X, X_, f=weightedtau, return_pvalues=False, **kwargs):
    """
    >>> ll = [[i] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> r = sortedness(a, b)
    >>> min(r), max(r)
    (-1.0, 0.998638259786)

    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> r = sortedness(a, b)
    >>> r
    array([ 0.24691868, -0.17456491,  0.19184376, -0.18193532,  0.07175694,
            0.27992254,  0.04121859,  0.16249574, -0.03506842,  0.27856259,
            0.40866965, -0.07617887,  0.12184064,  0.24762942, -0.05049511,
           -0.46277399,  0.12193493])
    """
    result, pvalues = [], []
    dist_matrix_X = cdist(X, X, metric="sqeuclidean")
    dist_matrix_X_ = cdist(X_, X_, metric="sqeuclidean")
    n_points = len(X)
    # Mask to remove diagonal.
    nI = ~eye(n_points, dtype=bool)
    # scores = -ranks for f=weightedtau.
    scores_X = -dist_matrix_X[nI].reshape(n_points, -1)
    scores_X_ = -dist_matrix_X_[nI].reshape(n_points, -1)
    for i in range(len(scores_X_)):
        corr, pvalue = f(scores_X[i, :], scores_X_[i, :], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result
