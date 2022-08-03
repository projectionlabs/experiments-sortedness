import matplotlib.pyplot as plt
from openml.datasets import get_dataset
from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from hoshmap import idict
from shelchemy.cache import sopen
from sortedness.kruskal import kruskal
from sortedness.local import ushaped_decay_f, sortedness
from sortedness.rank import rank_by_distances
from sortedness.trustworthiness import trustworthiness, continuity

from experimentssortedness.config import local_cache_uri, remote_cache_uri
from experimentssortedness.evaluation.plot import Plot, colors

functions = {
    "-1*kruskal": lambda X, X_: -kruskal(X, X_),
    "-1*nonmetric_kruskal": lambda X, X_: -kruskal(X, X_, f=rank_by_distances),
    "trustworthiness": trustworthiness,
    "continuity": continuity
}
projectors = {
    "PCA": PCA(n_components=2).fit_transform,
    "TSNE": lambda seed, X: TSNE(n_components=2, random_state=seed).fit_transform(X)
}
corr_funcs = {
    "ρ": spearmanr,
    "tau": kendalltau,
    "wtau-x": weightedtau,
    "wtau-x²": lambda X, X_: weightedtau(X, X_, weigher=lambda x: 1 / (1 + x ** 2)),
    "wtau-Ushaped": lambda X, X_: weightedtau(X, X_, weigher=ushaped_decay_f(n=len(X))),
    "sortedness": None,
}


def fetch_asnumpy(dataset):
    print(f"Loading {dataset}...", end="\t", flush=True)
    X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
    X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
    if dataset == "abalone":
        X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
    print("loaded!")
    return X.to_numpy()


def prepare_experiment(d):
    f = lambda X, X_, coefname: sortedness(X, X_, f=corr_funcs[coefname])
    for coefname in corr_funcs:
        d["coefname"] = coefname
        d[coefname] = f  # Apply proposed f functions (but don't evaluate yet)
    d >>= functions  # Apply all f functions from literature (but don't evaluate yet)
    return d


def plot(proj_name, d):
    for xlabel in functions:
        p = Plot(d, proj_name, xlabel, "'sortedness'", legend=True, plt=plt)
        for slabel, color in zip(corr_funcs.keys(), colors):
            p << (slabel, color)
        p.finish()

    for xlabel in ["-1*kruskal", "-1*nonmetric_kruskal"]:
        p = Plot(d, proj_name, xlabel, "cont, trust", legend=True, plt=plt)
        for slabel, color in zip(list(functions.keys())[2:], colors[1:]):
            p << (slabel, color)
        p.finish()

    p = Plot(d, proj_name, "trustworthiness", "continuity", legend=False, plt=plt)
    p << ("continuity", "blue")
    p.finish()

    p = Plot(d, proj_name, "-1*kruskal", "-1*nonmetric_kruskal", legend=False, plt=plt)
    p << ("-1*nonmetric_kruskal", "green")
    p.finish()

    plt.show()


data = idict(seed=0)
for projection, fproject in projectors.items():
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        for name in ["abalone", "iris"]:
            data["dataset"] = name
            data["X"] = fetch_asnumpy  # Apply proposed fetch() (but don't evaluate yet)
            data["X_"] = fproject  # Apply proposed fproject() (but don't evaluate yet)

            data = prepare_experiment(data)
            data >>= [local, remote]  # Add caches (but don't send/fetch anything yet)
            # data.evaluate()  # Force evaluation of all fields

            plot(f"{name}: {projection}", data)
            data.show()
