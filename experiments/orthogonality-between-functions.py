import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from experimentssortedness.config import local_cache_uri, remote_cache_uri
from experimentssortedness.data import fetch_asnumpy
from experimentssortedness.evaluation.plot import Plot, colors
from experimentssortedness.temporary import sortedness, stress, pwsortedness, rsortedness, global_pwsortedness
from hoshmap import idict
from shelchemy.cache import sopen

projectors = {
    # "PCA": PCA(n_components=2).fit_transform,
    "TSNE": lambda seed, X: TSNE(n_components=2, random_state=seed).fit_transform(X)
}
baseline = {  # TODO: speed up other measures as well
    "-1*kruskal": lambda X, X_: -stress(X, X_),
    # "trustworthiness": trustworthiness,
    # "continuity": continuity,
    "sortedness": lambda X, X_: sortedness(X, X_),
    "pw-sortedness": lambda X, X_: pwsortedness(X, X_),
    "rsortedness": lambda X, X_: rsortedness(X, X_),
}


def f(X, X_):
    r = [global_pwsortedness(X, X_)[0]] * X.shape[0]
    print(r)
    return r


alternative = {
    "sortedness": lambda X, X_: sortedness(X, X_),
    "pw-sortedness": lambda X, X_: pwsortedness(X, X_),
    "rsortedness": lambda X, X_: rsortedness(X, X_),
    "global pw sortedness": f,
}


def plot(proj_name, d):
    for xlabel in baseline:
        p = Plot(d, proj_name, xlabel, ylabel="'alternative'", legend=True, plt=plt)
        for slabel, color in zip(alternative.keys(), colors):
            print(slabel)
            p << (slabel, color)
        p.finish()
    plt.show()


data = idict(seed=0)
for projection, fproject in projectors.items():
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        for name in ["abalone"]:  # ["abalone", "iris"]:
            data["dataset"] = name
            data["X"] = fetch_asnumpy
            data["X_"] = fproject
            data = data >> baseline >> alternative
            print("AAAAAAAAAAAAAAAAAAAd cache")
            # data >>= [local]
            data.show()
            print(data.X.shape, data.X_.shape)
            plot(f"{name}: {projection}", data)
