import seaborn as sns
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import ap, gp
from numpy import mean, vstack
from numpy.random import normal, default_rng
from pandas import DataFrame
from sortedness.trustworthiness import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness
import matplotlib.font_manager as fm

print("Intended to show how measures behave with increasing shuffling.")
rng = default_rng(seed=0)

xmax, ymax, n = 100, 100, 1000
levels = [3.125, 6.25, 12.5, 25, 50, 100]
k = 5


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return trustworthiness(X, X_, k)


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: rsortedness(X, X_),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: sortedness(X, X_),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: pwsortedness(X, X_),
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - stress(X, X_),

}

xlabel = "Shuffling Level (\\%)"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, xmax, n)
y = rnd.uniform(0, ymax, n)
X = vstack((x, y)).T

xmin = min(X[:, 0])
xmax = max(X[:, 0])
ymin = min(X[:, 1])
ymax = max(X[:, 1])
indices = rng.choice(len(X), size=len(X), replace=False)
replacement = np.random.rand(len(indices), 2)
replacement[:, 0] = xmin + replacement[:, 0] * (xmax - xmin)
replacement[:, 1] = ymin + replacement[:, 1] * (ymax - ymin)


def randomize_projection(M, pct):
    get = int((len(M) * pct) // 100)
    projection_rnd = X.copy()
    projection_rnd[indices[:get], :] = replacement[:get, :]
    return projection_rnd


lvs, ms, vs = [], [], []
for m, f in measures.items():
    print(m)
    for level in levels:
        print(level, end=" ")
        X_ = randomize_projection(X, level)
        lvs.extend([level] * len(X))
        ms.extend([m] * len(X))
        vs.extend(f(X, X_))

print("---------------------_")
_, ax = plt.subplots(figsize=(15, 5))
df = DataFrame({xlabel: lvs, "Measure": ms, "Value": vs})
# ax.set_title('Loss curve', fontsize=15)
plt.rcParams["font.size"] = 23
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(plt.rcParams["font.size"])
    # linestyle=style, lw=width, color=color, logy=False, logx=False, fontsize=plt.rcParams["font.size"])

sns.boxplot(ax=ax, width=0.7, y='Value', x=xlabel, data=df, palette=["blue", "orange", "gray", "red", "brown"], hue='Measure')
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 0.9), borderaxespad=0)
plt.tight_layout()
plt.show()
