from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sortedness.trustworthiness import trustworthiness, continuity
from scipy.stats import weightedtau, spearmanr, kendalltau
from statistics import mean

from experimentssortedness.temporary import sortedness


from sklearn.neighbors import NearestNeighbors

#########################################################################
#########################################################################

def NeighborhoodPreservation(X, X_, k):
    neigh_ori = NearestNeighbors(n_neighbors=k).fit(X)
    neigh_proj = NearestNeighbors(n_neighbors=k).fit(X_)

    dist_ori, index_ori = neigh_ori.kneighbors(X)
    dist_proj, index_proj = neigh_proj.kneighbors(X_)

    percents = []
    for i in range(X.shape[0]):
        percents.append(len(list(set(index_ori[i]).intersection(index_proj[i])))/k)

    return sum(percents)/X.shape[0]

#############################################

def randomize_projection(X_, p):
    rnd_index = np.random.permutation(X_.shape[0])
    rnd_size = int(X_.shape[0]*(p/100))
    rnd_index = rnd_index[0:rnd_size]
    
    rnd_x = np.random.rand(rnd_size)
    rnd_y = np.random.rand(rnd_size)

    projection_rnd = X_.copy()
    projection_rnd[rnd_index, 0] = projection_rnd[rnd_index, 0] + rnd_x
    projection_rnd[rnd_index, 1] = projection_rnd[rnd_index, 1] + rnd_y

    return projection_rnd

#############################################

def randomize_projections(X_):
    projections = [X_.copy()]
    for i in range(10, 60, 10):
        projections.append(randomize_projection(X_, i))
    return projections

#############################################

def run_experiment(X, X_):
    projections = randomize_projections(X_)

    results_tw = []
    results_np = []
    results_s = []

    for proj in projections:
        # Compute TW
        # results_tw.append(trustworthiness(X.to_numpy(), proj))
        results_tw.append(trustworthiness(X, proj))

        # Compute NP
        results_np.append(NeighborhoodPreservation(X, proj, 10))

        # Compute Sortedness
        s = []
        for coefname in [weightedtau, spearmanr, kendalltau, None]:
            # s.append(sortedness(X.to_numpy(), proj, f=coefname))
            s.append(sortedness(X, proj, f=coefname))
        results_s.append(s)

    tw_mean = []
    for tw in results_tw:
        tw_mean.append(mean(tw))

    print(results_np)

    np_mean = []
    for np_values in results_np:
        np_mean.append(mean(np_values))

    s_p_mean = []
    s_tau_mean = []
    s_wtau_x_mean = []
    sortedness_mean = []
    box_p = []
    box_tau = []
    box_wtau_x = []
    box_sorterdness = []
    for s in results_s:
        s_p_mean.append(mean(s[0]))
        s_tau_mean.append(mean(s[1]))
        s_wtau_x_mean.append(mean(s[2]))
        sortedness_mean.append(mean(s[3]))
        box_p.append(s[0])
        box_tau.append(s[1])
        box_wtau_x.append(s[2])
        box_sorterdness.append(s[3])

    df = pd.DataFrame({
        'Sortedness ρ': s_p_mean,
        'Sortedness tau': s_tau_mean,
        'Sortedness wtau-x': s_wtau_x_mean,
        'Sortedness': sortedness_mean,
        'TW': tw_mean,
        'NP': np_mean
    })
    ax = df.plot(kind='line')
    ax.set_xticklabels(['', '0%', '10%', '20%', '30%', '40%', '50%'])
    # ax.set_xticklabels([1,2,3,4,5,6], ['0%', '10%', '20%', '30%', '40%', '50%'])
    plt.show()

    fig = plt.figure()
    bp = plt.boxplot(box_p+box_tau+box_wtau_x+box_sorterdness+results_tw+results_np,  vert = 1, patch_artist = True)

    bp_colors = np.repeat([plt.cm.Pastel1(0), plt.cm.Pastel1(1), plt.cm.Pastel1(2), plt.cm.Pastel1(3), plt.cm.Pastel1(4), plt.cm.Pastel1(5)], [6, 6, 6, 6, 6, 6], axis=0)
    bp_list = []
    for i, bplot in enumerate(bp['boxes']):
        bplot.set(color='gray', linewidth=3)
        bplot.set(facecolor = bp_colors[i])
        bp_list.append(bplot)

    for i, whisker in enumerate(bp['whiskers']):
        whisker.set(color='gray', linewidth=3)

    for i, cap in enumerate(bp['caps']):
        cap.set(color='gray', linewidth=3)

    for i, median in enumerate(bp['medians']):
        median.set(color='gray', linewidth=3)

    plt.legend([bp_list[0], bp_list[6], bp_list[12], bp_list[18]], ['Sortedness - p', 'Sortedness - tau', 'Sortedness - wtau-x', 'TW'], loc='lower right')
    plt.xticks(
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
        , ['0%', '10%', '20%', '30%', '40%', '50%', '0%', '10%', '20%', '30%', '40%', '50%', '0%', '10%', '20%', '30%', '40%', '50%'
        , '0%', '10%', '20%', '30%', '40%', '50%', '0%', '10%', '20%', '30%', '40%', '50%', '0%', '10%', '20%', '30%', '40%', '50%']
    )

    plt.title("Boxplot", loc="center", fontsize=18)
    plt.xlabel("Quantidade de pontos aleatorizados")
    plt.ylabel("")
    plt.show()

#########################################################################
#########################################################################

# For reproducability of the results
np.random.seed(42)

# sample_size = 1000
sample_size = 100

mnist = fetch_openml('mnist_784')

t_size = sample_size/mnist.data.shape[0]

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=t_size, random_state=42, shuffle=False
)

X = X_test
y = y_test

# feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
# df = pd.DataFrame(X,columns=feat_cols)
# df['y'] = y
# df['label'] = df['y'].apply(lambda i: str(i))

# N = df.shape[0]
# rndperm = np.random.permutation(df.shape[0])
# df_subset = df.loc[rndperm[:N],:].copy()
# df_subset = df.copy()
# data_subset = df_subset[feat_cols].values

# # Exibe as imagens do Mnist
# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,15):
#     ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
#     ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
# plt.show()

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

# Exibe a projecao da tSNE
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )
# plt.show()

############################################################

# run_experiment(X.to_numpy(), tsne_results)
run_experiment(tsne_results, tsne_results)
