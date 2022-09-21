import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp
from numpy import mean, vstack
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness

k = 5
limit = 200
distance, std = 100000, 1


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, n_neighbors=k))


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal s.": lambda X, X_: mean(rsortedness(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~weighted pairwise s.": lambda X, X_: mean(pwsortedness(X, X_)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise s.": lambda X, X_: global_pwsortedness(X, X_)[0],
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
}

a, b, c = (-distance, 0), (0, 0), (distance, 0)
d = {"cluster size": [int(x) for x in gp[2, 2.35, ..., limit]]}
for m, f in measures.items():
    print(m)
    d[m] = []
    for n in d["cluster size"]:
        print(n)
        X, y = make_blobs(
            n_samples=3 * n, cluster_std=[std, std, std], centers=[a, b, c],
            n_features=2, random_state=1, shuffle=False
        )
        dx = np.array([[distance, 0]])
        X_ = vstack((X[:n, :], X[n:2 * n, :] + dx, X[2 * n:, :] - dx))
        d[m].append(f(X, X_))

df = pd.DataFrame(d)
df.set_index("cluster size").plot()
plt.show()
