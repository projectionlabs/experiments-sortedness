import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp
from numpy import mean, vstack
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from experimentssortedness.temporary import sortedness, rsortedness, stress, pwsortedness, global_pwsortedness, sortedness1

print("Intended to show which measures are tolerant to speed up by truncating neighborhood list.")
raise NotImplemented
limit = 100

measures = {
    # "$T_5$~~~~~~~~trustworthiness": tw,
    # "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    # # "$\\overline{\\lambda}_{\\tau_w}'$~~~~~~sortedness'": lambda X, X_: mean(sortedness1(X, X_)),
    # "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    # "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    # "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
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

xlabel = "Noise Amplitude"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, xmax, n)
y = rnd.uniform(0, ymax, n)
X = vstack((x, y)).T
D = np.clip(X - rnd.normal(X, stdev), -stdev, stdev)
pprint(D)

d = {xlabel: [x * stdev for x in ap[1, 2, ..., steps]]}
for m, f in measures.items():
    print(m)
    d[m] = []
    for i in range(len(d[xlabel])):
        print(i, end=" ")
        X_ = X + i * D
        # pprint(X_)
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
    df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=True, fontsize=plt.rcParams["font.size"])

plt.tight_layout()
plt.show()
