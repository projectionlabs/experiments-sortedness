from openml.datasets import get_dataset


def fetch_asnumpy(dataset):
    print(f"Loading {dataset}...", end="\t", flush=True)
    X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
    X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
    if dataset == "abalone":
        X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
    print("loaded!")
    return X.to_numpy()
