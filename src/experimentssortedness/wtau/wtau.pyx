#cython: boundscheck=False, wraparound=False, nonecheck=False
import cython
from cpython cimport bool
from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy.math cimport INFINITY
from numpy.math cimport NAN
from numpy cimport ndarray, int64_t, float64_t, intp_t
import warnings
import numpy as np
cimport scipy.special.cython_special as cs
from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free

np.import_array()

ctypedef fused ordered:
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

cpdef float64_t weigh(  intp_t offset, 
                        intp_t length, 
                        intp_t iii, 
                        intp_t[::1] rank, 
                        intp_t[::1] perm, 
                        intp_t[::1] temp, 
                        float64_t[::1] exchanges_weights, 
                        ordered[::1] x, 
                        ordered[::1] y
                    ) nogil:
    cdef intp_t length0, length1, middle, i, j, k
    cdef float64_t weight, residual

    if length == 1:
        return 1./(1 + rank[perm[offset]])
    length0 = length // 2
    length1 = length - length0
    middle = offset + length0
    residual = weigh(offset, length0, iii, rank, perm, temp, exchanges_weights, x, y)
    weight = weigh(middle, length1, iii, rank, perm, temp, exchanges_weights, x, y) + residual
    if y[perm[middle - 1]] < y[perm[middle]]:
        return weight

    # merging
    i = j = k = 0

    while j < length0 and k < length1:
        if y[perm[offset + j]] <= y[perm[middle + k]]:
            temp[i] = perm[offset + j]
            residual -= 1./(1 + rank[temp[i]])
            j += 1
        else:
            temp[i] = perm[middle + k]
            exchanges_weights[iii] += 1./(1 + rank[temp[i]]) * (length0 - j) + residual
            k += 1
        i += 1

    perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
    perm[offset:offset+i] = temp[0:i]
    return weight


cpdef float64_t _weightedrankedtau( ordered[::1] x, ordered[::1] y, intp_t[:,::1] ranks, 
                                    int64_t iii, 
                                    int64_t n, 
                                    intp_t[:,::1] perms,
                                    intp_t[:,::1] temps,
                                    float64_t[::1] exchanges_weights, 
                                ) nogil:
    cdef ordered[::1] y_local = y
    cdef intp_t i, first
    cdef float64_t t, u, v, w, s, sq, tot, tau

    # if rank is None:
    #     # To generate a rank array, we must first reverse the permutation
    #     # (to get higher ranks first) and then invert it.
    #     rank = np.empty(n, dtype=np.intp)
    #     rank[...] = perm[::-1]
    #     _invert_in_place(rank)

    cdef intp_t[::1] perm = perms[iii,:]
    cdef intp_t[::1] temp = temps[iii,:]
    cdef intp_t[::1] rank = ranks[iii,:]

    # weigh joint ties
    first = 0
    t = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    t += s * (n - first - 1)

    # weigh ties in x
    first = 0
    u = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]]:
            u += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    u += s * (n - first - 1)
    # if first == 0: # x is constant (all ties)
    #     return np.nan

    # weigh discordances
    weigh(0, n, iii, rank, perm, temp, exchanges_weights, x, y)

    # weigh ties in y
    first = 0
    v = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if y[perm[first]] != y[perm[i]]:
            v += s * (i - first - 1) 
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    v += s * (n - first - 1) 
    # if first == 0: # y is constant (all ties)
    #     return np.nan

    # weigh all pairs
    s = sq = 0
    for i in range(n):
        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    tot = s * (n - 1) 

    tau = ((tot - (v + u - t)) - 2. * exchanges_weights[iii]) / sqrt(tot - u) / sqrt(tot - v)
    return min(1., max(-1., tau))




def parwtau(ordered[::1] x, ordered[::1] y, intp_t[:,::1] ranks):
    cdef int64_t n = np.int64(len(x))
    cdef int64_t m = np.int64(len(ranks))
    cdef float64_t[::1] ret = np.empty(m, dtype=np.float64)

    cdef intp_t[::1] lex = np.lexsort((y, x))
    cdef intp_t[:,::1] perms = np.zeros((m,n), dtype=np.intp)
    cdef intp_t[:,::1] temps = np.empty((m,n), dtype=np.intp)
    
    for ii in range(m):
        perms[ii] = lex
    cdef float64_t[::1] exchanges_weights = np.zeros(n, dtype=np.float64)

    cdef int64_t i, r
    for i in prange(m, nogil=True):
        # printf("%d\n", i)
    # for i in range(m):
        # with gil:
        # print(i, ranks.shape)
        ret[i] = _weightedrankedtau(x, y, ranks, i, n, perms, temps, exchanges_weights)
    return ret


cimport cython
import numpy as np
cimport openmp
from libc.math cimport log
from cython.parallel cimport prange
from cython.parallel cimport parallel

THOUSAND = 1024
FACTOR = 100
NUM_TOTAL_ELEMENTS = FACTOR * THOUSAND * THOUSAND
X1 = -1 + 2*np.random.rand(NUM_TOTAL_ELEMENTS)
X2 = -1 + 2*np.random.rand(NUM_TOTAL_ELEMENTS)
Y = np.zeros(X1.shape)

def test_serial():
    serial_loop(X1,X2,Y)

def serial_loop(double[:] A, double[:] B, double[:] C):
    cdef int N = A.shape[0]
    cdef int i

    for i in range(N):
        C[i] = log(A[i]) * log(B[i])

def test_parallel():
    parallel_loop(X1,X2,Y)

def parallel_loop(double[:] A, double[:] B, double[:] C):
    cdef int N = A.shape[0]
    cdef int i

    with nogil:
        for i in prange(N, schedule='static'):
            C[i] = log(A[i]) * log(B[i])








cpdef float64_t _weightedrankedtau0( ordered[::1] x, ordered[::1] y, intp_t[::1] rank_, 
                                    int64_t n, 
                                    intp_t[::1] perm,
                                    intp_t[::1] temp,
                                    float64_t[::1] exchanges_weights, 
                                ) nogil:
    cdef ordered[::1] y_local = y
    cdef intp_t i, first
    cdef float64_t t, u, v, w, s, sq, tot, tau
    exchanges_weights[0] = 0

    # if rank is None:
    #     # To generate a rank array, we must first reverse the permutation
    #     # (to get higher ranks first) and then invert it.
    #     rank = np.empty(n, dtype=np.intp)
    #     rank[...] = perm[::-1]
    #     _invert_in_place(rank)

    cdef intp_t[::1] rank = rank_  #[:]

    # weigh joint ties
    first = 0
    t = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    t += s * (n - first - 1)

    # weigh ties in x
    first = 0
    u = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]]:
            u += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    u += s * (n - first - 1)
    # if first == 0: # x is constant (all ties)
    #     return np.nan

    # weigh discordances
    weigh0(0, n, rank, perm, temp, exchanges_weights, x, y)

    # weigh ties in y
    first = 0
    v = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if y[perm[first]] != y[perm[i]]:
            v += s * (i - first - 1) 
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    v += s * (n - first - 1) 
    # if first == 0: # y is constant (all ties)
    #     return np.nan

    # weigh all pairs
    s = sq = 0
    for i in range(n):
        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    tot = s * (n - 1) 

    tau = ((tot - (v + u - t)) - 2. * exchanges_weights[0]) / sqrt(tot - u) / sqrt(tot - v)
    return min(1., max(-1., tau))




cpdef float64_t weigh0(  intp_t offset, 
                        intp_t length,                   
                        intp_t[::1] rank, 
                        intp_t[::1] perm, 
                        intp_t[::1] temp, 
                        float64_t[::1] exchanges_weights, 
                        ordered[::1] x, 
                        ordered[::1] y
                    ) nogil:
    cdef intp_t length0, length1, middle, i, j, k
    cdef float64_t weight, residual

    if length == 1:
        return 1./(1 + rank[perm[offset]])
    length0 = length // 2
    length1 = length - length0
    middle = offset + length0
    residual = weigh0(offset, length0, rank, perm, temp, exchanges_weights, x, y)
    weight = weigh0(middle, length1, rank, perm, temp, exchanges_weights, x, y) + residual
    if y[perm[middle - 1]] < y[perm[middle]]:
        return weight

    # merging
    i = j = k = 0

    while j < length0 and k < length1:
        if y[perm[offset + j]] <= y[perm[middle + k]]:
            temp[i] = perm[offset + j]
            residual -= 1./(1 + rank[temp[i]])
            j += 1
        else:
            temp[i] = perm[middle + k]
            exchanges_weights[0] += 1./(1 + rank[temp[i]]) * (length0 - j) + residual
            k += 1
        i += 1

    perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
    perm[offset:offset+i] = temp[0:i]
    return weight
