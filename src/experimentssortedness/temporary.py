"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""

import numpy as np
from numpy import eye, mean, argsort, arange, indices
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist
from scipy.stats import weightedtau, rankdata
from sklearn.decomposition import PCA


def sortedness(X, X_, f=weightedtau, return_pvalues=False, **kwargs):
    """
    # TODO: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
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
    >>> min(r), max(r)
    (-0.462773990559, 0.408669653064)
    >>> round(mean(r), 12)
    0.070104521222
    """
    result, pvalues = [], []
    sqdist_X = sqdist_matrix(X)
    sqdist_X_ = sqdist_matrix(X_)

    # For f=weightedtau: scores = -ranks.
    scores_X = -remove_diagonal(sqdist_X)
    scores_X_ = -remove_diagonal(sqdist_X_)

    for i in range(len(X)):
        corr, pvalue = f(scores_X[i, :], scores_X_[i, :], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def sqdist_matrix(X):
    return cdist(X, X, metric="sqeuclidean")


def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


def rsortedness(X, X_, f=weightedtau, return_pvalues=False, **kwargs):
    """
    Recyprocal sortedness: consider the neighborhood realtionship the other way around.
    Might be good to assess the effect of a projection on hubness.
    Break ties by comparing distances. In case of a new tie, break it by lexicographical order.
    TODO: fix tie breaker, and create flag to do that

    >>> ll = [[i, ] for i in range(6)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> #r = rsortedness(a, b)
    >>> #min(r), max(r)
    (-0.707870893072, 0.962964134515)

    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> r = rsortedness(b, a)
    >>> r
    array([ 0.20340735, -0.28571498,  0.14498561, -0.24662663,  0.09057659,
            0.30037263,  0.01412381,  0.18679486, -0.13212791,  0.28692565,
            0.4465226 , -0.17871893,  0.2020813 ,  0.28089447, -0.16861842,
           -0.5131998 ,  0.0955369 ])
    >>> min(r), max(r)
    (-0.513199801882, 0.44652260028)
    >>> round(mean(r), 12)
    0.042777358461
    """
    result, pvalues = [], []
    # sqdist_X = sqdist_matrix(X)
    # sqdist_X_ = sqdist_matrix(X_)
    # ranks0_X = rankdata(sqdist_X, axis=0, method="average")
    # ranks0_X_ = rankdata(sqdist_X_, axis=0, method="average")
    # print("sqdist_X")
    # print(sqdist_X)
    # print("ranks0_X")
    # print(ranks0_X)
    #
    # # Make ranking over sorted matrix to be able to break ties by method="ordinal".
    # idxs_X = argsort(sqdist_X, axis=1, kind="stable")
    # idxs_X_ = argsort(sqdist_X_, axis=1, kind="stable")
    # print("idxs_X")
    # print(idxs_X)
    # print()
    #
    # sorted_ranks0_X = ranks0_X[np.arange(ranks0_X.shape[0])[:, None], idxs_X]
    # sorted_ranks0_X_ = ranks0_X_[np.arange(ranks0_X_.shape[0])[:, None], idxs_X_]
    # print("sorted_ranks0_X")
    # print(sorted_ranks0_X)
    # print("1111111111\n")
    #
    # ranks1_X = rankdata(sorted_ranks0_X, axis=1, method="ordinal")
    # ranks1_X_ = rankdata(sorted_ranks0_X, axis=1, method="ordinal")
    # print("ranks1_X")
    # print(ranks1_X)
    # print()
    #
    # # Revert indexing.
    # identity = indices((len(X), len(X)))[0].transpose()
    # rev_idxs_X = identity[np.arange(identity.shape[0])[:, None], idxs_X]
    # rev_idxs_X_ = identity[np.arange(identity.shape[0])[:, None], idxs_X_]
    # print("rev_idxs_X")
    # print(rev_idxs_X)
    # print()
    #
    # ranks_X = ranks1_X[np.arange(ranks1_X.shape[0])[:, None], rev_idxs_X]
    # ranks_X_ = ranks1_X_[np.arange(ranks1_X_.shape[0])[:, None], rev_idxs_X_]
    # print("ranks_X")
    # print(ranks_X)
    # print()
    #
    # # Discard self-neighbors and make scores from ranks for compatibility with f=weightedtau.
    # scores_X = -(ranks_X[:, 1:])
    # scores_X_ = -(ranks_X_[:, 1:])
    # print(scores_X)
    # print()

    sqdist_X = sqdist_matrix(X)
    sqdist_X_ = sqdist_matrix(X_)
    ranks_X = rankdata(sqdist_X, axis=0)
    ranks_X_ = rankdata(sqdist_X_, axis=0)
    # scores = -ranks for f=weightedtau.
    scores_X = -remove_diagonal(ranks_X)
    scores_X_ = -remove_diagonal(ranks_X_)

    for i in range(len(X)):
        corr, pvalue = f(scores_X[i, :], scores_X_[i, :], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def gsortedness(X, X_, f=weightedtau, return_pvalue=False, **kwargs):
    """
    Global (pairwise) sortedness

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)

    >>> gsortedness(original, original)
    1.0
    >>> gsortedness(original, projected2)
    1.0
    >>> gsortedness(original, projected1)
    0.733798945632
    >>> gsortedness(original, projected2[:, 1:])
    0.18462055725
    >>> gsortedness(original, projectedrnd)
    -0.050511068452
    """
    result, pvalues = [], []
    dists_X = pdist(X, metric="sqeuclidean")
    dists_X_ = pdist(X_, metric="sqeuclidean")

    # For f=weightedtau: scores = -ranks.
    scores_X = -dists_X
    scores_X_ = -dists_X_

    corr, pvalue = f(scores_X, scores_X_, **kwargs)
    result.append(round(corr, 12))
    pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalue:
        return np.array(list(zip(result, pvalues)))
    return result[0]
