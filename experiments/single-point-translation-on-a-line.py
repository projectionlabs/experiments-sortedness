from math import sqrt
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp, ap
from numpy import mean, vstack, hstack
from scipy.stats import kendalltau
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness, sortedness1

print("Intended to show sensitiveness (which measures is triggered earlier) and discontinuities for a monotonically increasing distortion.")
k = 5
n = 25
l = 2 * n


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, n_neighbors=k))


measures = {
    # "$T_5$~~~~~~~~trustworthiness": tw,
    # "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_, f=kendalltau)),
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    # "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_, f=kendalltau)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
    # "sortedness": lambda X, X_: mean(sortedness(X, X_, f=kendalltau)),
    # "$1-\\sigma_nm$~~nonmetric stress": lambda X, X_: 1 - mean(stress(X, X_, metric=False)),
    # "-gsortedness_w": lambda X, X_: gsortedness(X, X_, weigher=lambda r: r + 1),
    # "rsortedness": lambda X, X_: mean(rsortedness(X, X_, f=kendalltau)),
    # "continuity": lambda X, X_: mean(continuity(X, X_, k=15)),
    # "ρ": lambda X, X_: mean(sortedness(X, X_, f=spearmanr)),
    # "sortedness_x²": lambda X, X_: mean(sortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)))),
    # "rρ": lambda X, X_: mean(rsortedness(X, X_, f=spearmanr)),
    # "rsortedness_x²": lambda X, X_: mean(rsortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)))),
    # "gρ": lambda X, X_: gsortedness(X, X_, f=spearmanr),
    # "gsortedness_x²": lambda X, X_: gsortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2))),
    # "gsortedness_w": lambda X, X_: gsortedness(X, X_),
    # "$\\overline{\\lambda}_{\\tau_w}'$~~~~~~sortedness'": lambda X, X_: mean(sortedness1(X, X_)),
}

xlabel = "Offset"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, l, n)
x[0] = 0
y = np.zeros(n)
X = vstack((x, np.zeros(n))).T
X.sort(axis=0)
pprint(X)
X_ = X.copy()
d = {xlabel: [int(x) for x in ap[0, 1, ..., l - 2]]}
for m, f in measures.items():
    print(m)
    d[m] = []
    for e in d[xlabel]:
        print(e, sep=" ")
        X_[0, 0] = e
        d[m].append(f(X, X_))
        # r = f(X, X_)
        # print(min(r), max(r))
        # print(r.tolist())
    print(d)

df = pd.DataFrame(d)
df.set_index(xlabel).plot()
plt.show()
