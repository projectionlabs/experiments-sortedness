from openml.datasets import get_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def fetch_asnumpy(dataset, sample=500):
    print(f"Loading {dataset}...", end="\t", flush=True)
    X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
    X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
    if dataset == "abalone":
        X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
    print("loaded!")
    return X.to_numpy()[:sample]


def fetch(name, sample=500):
    d = fetch_openml(name)
    return d.data[:sample], d.target[:sample]


def split(data, target, sample_size):
    t_size = sample_size / data.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=t_size, random_state=42, shuffle=False
    )
    return X_train, X_test
