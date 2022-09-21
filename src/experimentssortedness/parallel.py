from itertools import chain

import numba
import numpy as np
from numba import prange, set_num_threads
from scipy.spatial.distance import sqeuclidean, squareform
from scipy.stats import rankdata
import pathos.multiprocessing as mp

from shelchemy.lazy import ichunks


def rankcol(X, method="average"):
    step = 1000
    jobs = mp.ThreadingPool().imap(lambda M: rankdata(M, axis=0, method=method), (X[:, j:j + step] for j in range(0, len(X), step)))
    return np.hstack(list(jobs))


set_num_threads(16)


@numba.jit(nopython=True, parallel=True, cache=True)
def pw_sqeucl_nb(M, M_):
    n = len(M)
    m = (n ** 2 - n) // 2
    scores = np.zeros(m)
    scores_ = np.zeros(m)
    for i in prange(n):
        num = 2 * n * i - i ** 2 - i
        for j in range(i + 1, n):
            c = num // 2 + j - i - 1
            sub = M[i] - M[j]
            scores[c] = -np.dot(sub, sub)
            sub_ = M_[i] - M_[j]
            scores_[c] = -np.dot(sub_, sub_)
    return scores, scores_


def pw_sqeucl(M):
    n = len(M)
    li = (M[i] for i in range(n) for j in range(i + 1, n))
    lj = (M[j] for i in range(n) for j in range(i + 1, n))
    jobs = mp.ThreadingPool().imap(lambda l,g: [sqeuclidean(a, b) for a, b in zip(l,g)], ichunks(li, 20, asgenerators=False), ichunks(lj, 20, asgenerators=False))
    return np.array(list(chain(*jobs)))
