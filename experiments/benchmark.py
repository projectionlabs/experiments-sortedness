from timeit import timeit

import numpy as np
from numpy import eye
from sklearn.decomposition import PCA

from experimentssortedness.temporary import pwsortedness, rsortedness, global_pwsortedness, stress, sortedness

ffs = [stress, sortedness, rsortedness, pwsortedness, global_pwsortedness]
nns = ["stress        ", "sortedness  ", "rsortedness  ", "pwsortedness ", "global_pwsort."]
for n in range(1000, 1001):
    d = n // 5
    m = (0,) * d
    cov = eye(d)
    rng = np.random.default_rng(seed=0)
    original = rng.multivariate_normal(m, cov, size=n)
    projected1 = PCA(n_components=n // 10).fit_transform(original)
    r = [0, 0]
    print(original.size)
    for label, ff in list(zip(nns, ffs)):
        def f():
            ff(original, projected1)
        print(label, n, d, timeit(f, number=1), sep="\t")
