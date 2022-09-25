"""
This file has the updated version of the proposed measure sortedness.
It is here to help fixing bugs, but it will be moved to the package when done.
"""
import gc

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, mean, sqrt, minimum, argsort
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata, kendalltau, weightedtau
from sklearn.decomposition import PCA

from experimentssortedness.matrices import index
from experimentssortedness.parallel import rank_alongrow, rank_alongcol
from experimentssortedness.wtau import parwtau
from shelchemy.lazy import ichunks


def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


def sortedness(X, X_, f=weightedtau, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
    # TODO: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        values can be lower due to that; including perfect projections.
        that is, avoid penalizing a correct projection because of a tie
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
    if parallel_kwargs is None:
        parallel_kwargs = {}
    result, pvalues = [], []
    npoints = len(X)

    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    sqdist_X, sqdist_X_ = tmap(lambda M: cdist(M, M), [X, X_])

    # For f=weightedtau: scores = -ranks.
    scores_X = -remove_diagonal(sqdist_X)
    scores_X_ = -remove_diagonal(sqdist_X_)

    # TODO: Offer option to use parwtau (when/if it becomes a working thing)
    for i in range(len(X)):
        corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def sortedness1(X, X_, f=weightedtau, return_pvalues=False, **kwargs):
    """
    sortedness' (lambda'𝜏w)

    # TODO: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        values can be lower due to that; including perfect projections.
        that is, avoid penalizing a correct projection because of a tie
        [here we are breaking ties by lexicographical order, differently from other functions in the file]
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

    >>> r = sortedness1(original, original)
    >>> min(r), max(r), round(mean(r), 12)
    (1.0, 1.0, 1.0)
    >>> r = sortedness1(original, projected2)
    >>> min(r), round(mean(r), 12), max(r)
    (1.0, 1.0, 1.0)
    >>> r = sortedness1(original, projected1)
    >>> min(r), round(mean(r), 12), max(r)
    (0.100283116914, 0.324507531866, 0.621992330757)
    >>> r = sortedness1(original, projected2[:, 1:])
    >>> min(r), round(mean(r), 12), max(r)
    (-0.508506647872, 0.122451450028, 0.704705474788)
    >>> r = sortedness1(original, projectedrnd)
    >>> min(r), round(mean(r), 12), max(r)
    (-0.216685979143, 0.073607212115, 0.426596265724)
    """
    n = len(X)
    result, pvalues = [], []
    sqdist_X = cdist(X, X)
    sqdist_X_ = cdist(X_, X_)
    D = remove_diagonal(sqdist_X)
    D_ = remove_diagonal(sqdist_X_)

    # Calculate indexing and sorting for sector widths.
    idx = argsort(D, kind="stable", axis=1)
    S, S_ = index(D, idx), index(D_, idx)
    S[:, 1:] -= S[:, :-1]
    S_[:, 1:] -= S_[:, :-1]

    # R, R_ = rankdata(S, axis=1)as, rankdata(S_, axis=1)astype
    # c = 2*(1 - count_nonzero(R-R_, axis=1) / n) - 1
    # return c

    # Take negative widths to be used as scores.
    scores = -S
    scores_ = -S_

    for i in range(n):
        corr, pvalue = f(scores[i], scores_[i], rank=idx[i], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def rsortedness(X, X_, f=weightedtau, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
    Reciprocal sortedness: consider the neighborhood realtion the other way around.
    Might be good to assess the effect of a projection on hubness, and also to serve as a loss function for a custom projection algorithm.
    Break ties by comparing distances. In case of a new tie, break it by lexicographical order.
    # TODO: add flag to break (not so rare cases of) ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        values can be lower due to that; including perfect projections.
        that is, avoid penalizing a correct projection because of a tie

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
    array([ 0.20340735, -0.28996097,  0.14889235, -0.27620087,  0.09057659,
            0.2807414 ,  0.02338712,  0.21883492, -0.13212791,  0.27294288,
            0.43250004, -0.19405479,  0.17832715,  0.26649029, -0.16861842,
           -0.44989766,  0.16413694])
    >>> min(r), max(r), round(mean(r), 12)
    (-0.449897660668, 0.4325000425, 0.045257436362)
    """
    if parallel_kwargs is None:
        parallel_kwargs = {}
    npoints = len(X)
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    D, D_ = tmap(lambda M: cdist(M, M, metric="sqeuclidean"), [X, X_])
    R, R_ = (rank_alongcol(M, parallel=parallel) for M in [D, D_])
    scores_X, scores_X_ = tmap(lambda M: -remove_diagonal(M), [R, R_])  # For f=weightedtau: scores = -ranks.
    # tmap = pmap
    # pmap = tmap
    if hasattr(f, "isparwtau"):
        raise Exception("disagree with other version")
        return parwtau(scores_X, scores_X_, npoints, parallel=parallel, **kwargs)

    def thread(l):
        lst1 = []
        lst2 = []
        for i in l:
            corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
            lst1.append(round(corr, 12))
            lst2.append(round(pvalue, 12))
        return lst1, lst2

    result, pvalues = [], []
    jobs = pmap(thread, ichunks(range(npoints), 15, asgenerators=False))
    for corrs, pvalues in jobs:
        result.extend(corrs)
        pvalues.extend(pvalues)

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def global_pwsortedness(X, X_, parallel=True, parallel_n_trigger=10000, **parallel_kwargs):
    """
    Global pairwise sortedness (Λ𝜏1)

    # TODO: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        values can be lower due to that; including perfect projections.
        that is, avoid penalizing a correct projection because of a tie

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
    npoints = len(X)
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    dists_X, dists_X_ = tmap(thread, [X, X_])
    return kendalltau(dists_X, dists_X_)


def pwsortedness(X, X_, rankings=None, parallel=True, parallel_n_trigger=200, batches=10, debug=False, **parallel_kwargs):
    """
    Local pairwise sortedness (Λ𝜏w)

    # TODO: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        this probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        values can be lower due to that; including perfect projections.
        that is, avoid penalizing a correct projection because of a tie

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
        Should be a 2d numpy array with one ranking per column. Each row represents a pair.
        None: calculate internally based on proximity of each pair to the point of interest
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
    (0.646666346009, 0.761682319455, 0.831788803323)
    >>> r = pwsortedness(original, projected2[:, 1:])
    >>> min(r), round(mean(r), 12), max(r)
    (0.160377881152, 0.205647446418, 0.265690049487)
    >>> r = pwsortedness(original, projectedrnd)
    >>> min(r), round(mean(r), 12), max(r)
    (-0.144615079553, -0.106565866692, -0.072703466403)
    """
    npoints = len(X)
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    if debug: print(1)
    thread = lambda M: -pdist(M, metric="sqeuclidean")
    scores_X, scores_X_ = tmap(thread, [X, X_])
    if debug: print(2)
    if rankings is None:
        D, D_ = tmap(squareform, [scores_X, scores_X_])
        if debug: print(3)
        n = len(D)
        m = (n ** 2 - n) // 2
        M = np.zeros((m, n))
        if debug: print(4)
        c = 0
        for i in range(n - 1):  # a bit slow, but only a fraction of wtau (~5%)
            h = n - i - 1
            d = c + h
            M[c:d] = minimum(D[i] + D[i + 1:], D_[i] + D_[i + 1:])
            c = d
        if debug: print(5)
        del D
        del D_
        gc.collect()
        if debug: print(6)
        M = rank_alongrow(M.T, step=n // batches, parallel=parallel, **parallel_kwargs).T
        if debug: print(7)
        # M = rank_alongcol(M, step=n // batches)  # Slower and RAM heavier.
    else:
        M = rankings
    return np.round(parwtau(scores_X, scores_X_, npoints, M, parallel=parallel, **parallel_kwargs), 12)


def stress(X, X_, metric=True, parallel=True, parallel_size_trigger=10000, **parallel_kwargs):
    """
    Kruskal's "Stress Formula 1" normalized before comparing distances.
    default: Euclidean

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = stress(original, original*5)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.073462710191, 0.333885390367, array([0.2812499 , 0.31103416, 0.21994448, 0.07346271, 0.2810867 ,
           0.16411944, 0.17002148, 0.14748528, 0.18341208, 0.14659984,
           0.33388539, 0.24110857]))
    >>> stress(original, projected)
    array([0.2812499 , 0.31103416, 0.21994448, 0.07346271, 0.2810867 ,
           0.16411944, 0.17002148, 0.14748528, 0.18341208, 0.14659984,
           0.33388539, 0.24110857])
    >>> stress(original, projected, metric=False)
    array([0.33947258, 0.29692937, 0.30478874, 0.10509128, 0.2516135 ,
           0.2901905 , 0.1662822 , 0.13153341, 0.34299717, 0.164696  ,
           0.35266095, 0.35276684])



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
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and X.size > parallel_size_trigger else map
    # TODO: parallelize cdist in slices?
    if metric:
        thread = lambda M, m: cdist(M, M, metric=m)
        Dsq, D_ = tmap(thread, [X, X_], ["sqeuclidean", "Euclidean"])  # Slowest part (~98%).
        Dsq /= np.max(Dsq)
        D = sqrt(Dsq)
        D_ /= np.max(D_)
    else:
        thread = lambda M: rankdata(cdist(M, M, metric="sqeuclidean"), method="average", axis=1) - 1
        D, D_ = tmap(thread, [X, X_])
        Dsq = D ** 2

    sqdiff = (D - D_) ** 2
    nume = sum(sqdiff)
    deno = sum(Dsq)
    result = np.round(sqrt(nume / deno), 12)
    return result
