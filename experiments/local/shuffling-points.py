from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import ap
from numpy import mean, vstack
from numpy.random import normal, default_rng
from sortedness.trustworthiness import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness
import matplotlib.font_manager as fm

print("Intended to show how measures behave with increasing shuffling.")
rng = default_rng()


def randomize_projection(X_, pct):
    xmin = min(X_[:, 0])
    xmax = max(X_[:, 0])
    ymin = min(X_[:, 1])
    ymax = max(X_[:, 1])
    indices = rng.choice(len(X_), size=int((len(X_) * pct) // 100), replace=False)
    projection_rnd = X_.copy()
    replacement = np.random.rand(len(indices), 2)
    replacement[:, 0] = xmin + replacement[:, 0] * (xmax - xmin)
    replacement[:, 1] = ymin + replacement[:, 1] * (ymax - ymin)
    projection_rnd[indices] = replacement
    return projection_rnd


xmax, ymax, n = 100, 100, 1000
levels = [0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k = 5


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, k))


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
}

xlabel = "Shuffling Level (\\%)"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, xmax, n)
y = rnd.uniform(0, ymax, n)
X = vstack((x, y)).T

d = {xlabel: levels}
for m, f in measures.items():
    print(m)
    d[m] = []
    for level in d[xlabel]:
        print(level, end=" ")
        X_ = randomize_projection(X, level)
        d[m].append(f(X, X_))
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
