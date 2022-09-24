from itertools import chain

import numpy as np
import pathos.multiprocessing as mp
from scipy.spatial.distance import sqeuclidean
from scipy.stats import rankdata

from shelchemy.lazy import ichunks


def rank_alongcol(X, method="average", step=10):
    n = len(X)
    if step > n:
        step = n
    it = (X[:, j:j + step] for j in range(0, n, step))
    jobs = mp.ThreadingPool().imap(lambda M: rankdata(M, axis=0, method=method), it)
    return np.hstack(list(jobs)).astype(int) - 1


def rank_alongrow(X, method="average", step=10):
    n = len(X)
    if step > n:
        step = n
    it = (X[j:j + step] for j in range(0, n, step))
    jobs = mp.ThreadingPool().imap(lambda M: rankdata(M, axis=1, method=method), it)
    return np.vstack(list(jobs)).astype(int) - 1


# set_num_threads(16)


# @numba.jit(nopython=True, parallel=True, cache=True)
# def pw_sqeucl_nb(M, M_):
#     n = len(M)
#     m = (n ** 2 - n) // 2
#     scores = np.zeros(m)
#     scores_ = np.zeros(m)
#     for i in prange(n):
#         num = 2 * n * i - i ** 2 - i
#         for j in range(i + 1, n):
#             c = num // 2 + j - i - 1
#             sub = M[i] - M[j]
#             scores[c] = -np.dot(sub, sub)
#             sub_ = M_[i] - M_[j]
#             scores_[c] = -np.dot(sub_, sub_)
#     return scores, scores_


def pw_sqeucl(M):
    n = len(M)
    li = (M[i] for i in range(n) for j in range(i + 1, n))
    lj = (M[j] for i in range(n) for j in range(i + 1, n))
    jobs = mp.ThreadingPool().imap(lambda l, g: [sqeuclidean(a, b) for a, b in zip(l, g)], ichunks(li, 20, asgenerators=False), ichunks(lj, 20, asgenerators=False))
    return np.array(list(chain(*jobs)))
