import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def neighborhood_preservation(X, X_, k):
    neigh_ori = NearestNeighbors(n_neighbors=k).fit(X)
    neigh_proj = NearestNeighbors(n_neighbors=k).fit(X_)

    dist_ori, index_ori = neigh_ori.kneighbors(X)
    dist_proj, index_proj = neigh_proj.kneighbors(X_)

    percents = []
    for i in range(X.shape[0]):
        percents.append(len(list(set(index_ori[i]).intersection(index_proj[i])))/k)
    
    return sum(percents)/X.shape[0]

######################################################

# For reproducability of the results
np.random.seed(42)

sample_size = 100

# mnist = fetch_openml('mnist_784')
mnist = load_digits()

t_size = sample_size/mnist.data.shape[0]

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=t_size, random_state=42, shuffle=False
)

X = X_test
y = y_test

print(neighborhood_preservation(X, X, 10))