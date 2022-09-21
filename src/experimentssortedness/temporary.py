"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""
import gc
from itertools import repeat, chain

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, mean, sqrt, maximum, minimum
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform, sqeuclidean
from scipy.stats import weightedtau, rankdata, kendalltau
from sklearn.decomposition import PCA

from experimentssortedness.parallel import pw_sqeucl
from shelchemy.lazy import ichunks


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
    sqdist_X = cdist(X, X)
    sqdist_X_ = cdist(X_, X_)

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


def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


def rsortedness(X, X_, f=weightedtau, return_pvalues=False, parallel=False, **kwargs):
    """
    Reciprocal sortedness: consider the neighborhood realtion the other way around.
    Might be good to assess the effect of a projection on hubness, and also to serve as a loss function for a custom projection algorithm.
    Break ties by comparing distances. In case of a new tie, break it by lexicographical order.
    # TODO: add flag to break not so rare cases of ties that persist after projection (implies a much slower algorithm)
        this is needed to avoid penalizing a correct projection because of a tie

    >>> ll = [[i, ] for i in range(17)]
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
    >>> min(r), max(r), round(mean(r), 12)
    (-0.513199801882, 0.44652260028, 0.042777358461)
    """
    tmap = mp.ThreadingPool().imap if len(X) > 0 and parallel else map
    pmap = mp.ProcessingPool().imap if len(X) > 0 and (parallel or parallel is None) else map

    # TODO: check if parallelization really brings more benefits (CPU) than problems (RAM)
    def thread(M):
        D = cdist(M, M, metric="sqeuclidean")
        R = rankdata(D, axis=0)
        return -remove_diagonal(R)  # For f=weightedtau: scores = -ranks.

    scores_X, scores_X_ = pmap(thread, [X, X_])

    def thread(l):
        lst1 = []
        lst2 = []
        for i in l:
            corr, pvalue = f(scores_X[i, :], scores_X_[i, :], **kwargs)
            lst1.append(round(corr, 12))
            lst2.append(round(pvalue, 12))
        return lst1, lst2

    result, pvalues = [], []
    jobs = pmap(thread, ichunks(range(len(X)), 15, asgenerators=False))
    for corrs, pvalues in jobs:
        result.extend(corrs)
        pvalues.extend(pvalues)

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def global_pwsortedness(X, X_, parallel=True):
    """
    Global pairwise sortedness (Λ𝜏1)

    Parameters
    ----------
    X
        Original dataset or precalculated pairwise squared distances from pdist(X, metric="sqeuclidean")
    X_
        Projected points or precalculated pairwise squared distances from pdist(X, metric="sqeuclidean")

    Returns
    -------
    (Λ𝜏1, p-value)
        The p-value considers as null hypothesis the absence of order, i.e., Λ𝜏1 = 0.
    """
    # TODO: parallelize pdist into a for?
    thread = lambda M: pdist(M, metric="sqeuclidean")
    xmap = mp.ThreadingPool().imap if parallel else map
    dists_X, dists_X_ = xmap(thread, [X, X_])
    return kendalltau(dists_X, dists_X_)


def pwsortedness(X, X_, f=weightedtau, rankings=True, return_pvalues=False, parallel=True, **kwargs):
    """
    Local pairwise sortedness (Λ𝜏w)

    Parameters
    ----------
    X
        Original dataset
    X_
        Projected points
    f
        Ranking correlation function
    rankings
        External importance ranking for each point in relation to all others.
        Should be a 2d numpy array with one ranking per row.
        True: calculate internally based on proximity of pair centroid to the point of interest
        False: do not provide rankings to `f`
    return_pvalues
        Flag to include the second item returned by `f`, usually a p-value, in the resulting list
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    kwargs
        Any extra argument to be provided to `f`

    Returns
    -------
        Numpy vector or, if p-values are included, 2d array

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> m = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(m, cov, size=12)
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)

    >>> r = pwsortedness(original, original)
    >>> min(r), max(r), round(mean(r), 12)
    (1.0, 1.0, 1.0)
    >>> r = pwsortedness(original, projected2)
    >>> min(r), round(mean(r), 12), max(r)
    (1.0, 1.0, 1.0)
    >>> r = pwsortedness(original, projected1)
    >>> min(r), round(mean(r), 12), max(r)
    (0.705228292232, 0.769490925935, 0.812276273024)
    >>> r = pwsortedness(original, projected2[:, 1:])
    >>> min(r), round(mean(r), 12), max(r)
    (0.152876321633, 0.183289364037, 0.240046100113)
    >>> r = pwsortedness(original, projectedrnd)
    >>> min(r), round(mean(r), 12), max(r)
    (-0.130051112362, -0.094590524401, -0.065543415145)
    """
    pmap = mp.ProcessingPool().imap if parallel else map
    tmap = mp.ThreadingPool().imap if parallel else map
    thread = lambda M: -pdist(M, metric="sqeuclidean")
    scores_X, scores_X_ = pmap(thread, [X, X_])

    if rankings is True:
        D, D_ = tmap(squareform, [scores_X, scores_X_])
        n = len(D)
        m = (n ** 2 - n) // 2
        E = np.zeros((m, n))
        E_ = np.zeros((m, n))
        c = 0
        for i in range(n - 1):
            h = n - i - 1
            d = c + h
            E[c:d] = D[i] + D[i + 1:]
            E_[c:d] = D_[i] + D_[i + 1:]
            c = d
        del D
        del D_
        gc.collect()
        M = minimum(E, E_)
        del E
        del E_
        gc.collect()

    def thread(r):
        if rankings is True:
            corr, pvalue = f(scores_X, scores_X_, rank=r, **kwargs)
        elif rankings is False:
            corr, pvalue = f(scores_X, scores_X_, **kwargs)
        else:
            corr, pvalue = f(scores_X, scores_X_, rank=r, **kwargs)
        return round(corr, 12), round(pvalue, 12)

    result, pvalues = [], []
    lst = (rankdata(M[:, i]).astype(int) for i in range(len(X)))
    for corrs, pvalue in pmap(thread, lst):
        result.append(corrs)
        pvalues.append(pvalue)

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def stress(X, X_, metric=True, parallel=True, **kwargs):
    """
    Kruskal's "Stress Formula 1"
    default: Euclidean

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = stress(original, original)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.081106807792, 0.347563916162, array([0.29566817, 0.31959501, 0.23577467, 0.08110681, 0.29811345,
           0.18098479, 0.18240664, 0.155316  , 0.20012608, 0.15791188,
           0.34756392, 0.25626217]))
    >>> stress(original, projected)
    array([0.29566817, 0.31959501, 0.23577467, 0.08110681, 0.29811345,
           0.18098479, 0.18240664, 0.155316  , 0.20012608, 0.15791188,
           0.34756392, 0.25626217])
    >>> stress(original, projected, metric=False)
    array([0.28449947, 0.25842568, 0.24773341, 0.09558354, 0.22450148,
           0.23819653, 0.15127226, 0.1167218 , 0.2901905 , 0.14607233,
           0.31265683, 0.29262076])



    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    metric
        Stress formula version: metric or nonmetric
    parallel
        Parallelize processing when |X|>1000. Might use more memory.

    Returns
    -------

    """
    xmap = mp.ThreadingPool().imap if parallel and len(X) > 1000 else map
    # TODO: parallelize cdist in slices?
    if metric:
        thread = lambda M, m: cdist(M, M, metric=m)
        Dsq, D_ = xmap(thread, [X, X_], ["sqeuclidean", "Euclidean"])
        D = sqrt(Dsq)
    else:
        thread = lambda M: rankdata(cdist(M, M, metric="sqeuclidean"), method="average", axis=1)
        D, D_ = xmap(thread, [X, X_])
        Dsq = D ** 2

    sqdiff = (D - D_) ** 2
    nume = sum(sqdiff)
    deno = sum(Dsq)
    result = np.round(sqrt(nume / deno), 12)
    return result
