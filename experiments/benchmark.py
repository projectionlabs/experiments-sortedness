from timeit import timeit

import numpy as np
from numpy import eye, mean
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA

from experimentssortedness.temporary import pwsortedness, rsortedness, global_pwsortedness, stress

n = 1000
d = n // 2
m = (0,) * d
cov = eye(d)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(m, cov, size=n)
projected1 = PCA(n_components=n // 3).fit_transform(original)

r = [0, 0]


def f():
    r[0] = pwsortedness(original, projected1, parallel=True)
    return r[0]


print("pyx  ", timeit(f, number=1), sep="\t")
# print("scipy", timeit(g, number=1), sep="\t")
# assert_allclose(r[0], r[1])
print()

# print(timeit(lambda: pwsortedness(original, projected1, parallel=False), number=1))
# # exit()
# print()
# # print(timeit(lambda: stress(original, projected1, parallel=False), number=1))
# # print(timeit(lambda: stress(original, projected1, parallel=True), number=1))
# a = pwsortedness(original[:100], projected1[:100], parallel=True)
# b = pwsortedness(original[:100], projected1[:100], parallel=False)
# print((a == b).all())

exit()


# D = cdist(original, original, metric="sqeuclidean")

# l1 = [round(timeit(lambda: rankdata(D, axis=0), number=1), 1) for _ in range(3)]
# l2 = [round(timeit(lambda: rankcol(D), number=1), 1) for _ in range(3)]
# print(min(l1), max(l1))
# print(min(l2), max(l2))
# a = rankdata(D, axis=0)
# b = rankcol(D)
# print(a,b,sep="\n")
# print((a == b).all())


def f(*args, **kwargs):
    return pwsortedness(*args, **kwargs)
    # return global_pwsortedness(*args, **kwargs)
    # return stress(*args, **kwargs)


print((f(original, projected1, parallel=False) == f(original, projected1, parallel=True)).all())
# print(timeit(lambda: pwsortedness(original, projected1), number=1))
print(timeit(lambda: f(original, projected1, parallel=False), number=1))
print(timeit(lambda: f(original, projected1, parallel=True), number=1))
# print(timeit(lambda: f(original, projected1, parallel=None), number=1))
# print(pwsortedness(original, projected1)==pwsortedness2(original, projected1))
#
# print(f(original, projected1, parallel=None))
# print(f(original, projected1, parallel=True))
# print(f(original, projected1, parallel=False))
