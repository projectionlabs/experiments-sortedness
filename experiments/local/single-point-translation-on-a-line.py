from math import sqrt
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp, ap
from numpy import mean, vstack, hstack
from scipy.stats import kendalltau
from sklearn.datasets import make_blobs
from sortedness.trustworthiness import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness, sortedness1

print("Intended to show sensitiveness (which measures is triggered earlier) and discontinuities for a monotonically increasing distortion.")
k = 5
n = 25
l = 2 * n


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return trustworthiness(X, X_, k)[0]


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: (rsortedness(X, X_))[0],
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: (sortedness(X, X_))[0],
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: (pwsortedness(X, X_))[0],
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - (stress(X, X_))[0],
}

xlabel = "Offset"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, l, n)
x[0] = 0
y = np.zeros(n)
X = vstack((x, y)).T
X.sort(axis=0)
pprint(X)
X_ = X.copy()
d = {xlabel: [int(x) for x in ap[1, 2, ..., l]]}
for m, f in measures.items():
    print(m)
    d[m] = []
    for e in d[xlabel]:
        print(e, sep=" ")
        X_[0, 0] = e
        d[m].append(f(X, X_))
        pprint(X_)
    print(d)

print("---------------------_")
_, ax = plt.subplots(figsize=(15, 5))
df = pd.DataFrame(d)
df = df.set_index(xlabel)  # .plot()
# ax.set_title('Loss curve', fontsize=15)
plt.rcParams["font.size"] = 23
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(plt.rcParams["font.size"])
for (ylabel, data), (style, width, color) in zip(list(d.items())[1:], [
    ("dotted", 1.5, "blue"),
    ("dotted", 3, "orange"),
    ("dotted", 3, "black"),
    ("-.", 3, "red"),
    ("dashed", 3, "purple"),
    ("dashed", 1.5, "brown"),
]):
    print("\n" + ylabel)
    df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=False, fontsize=plt.rcParams["font.size"])

plt.grid()
plt.tight_layout()
plt.show()
