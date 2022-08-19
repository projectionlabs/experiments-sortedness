from lange import ap
from numpy.random import default_rng
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
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

#########################################################################
#########################################################################

rng = default_rng()


def randomize_projection(X_, p):
    xmin = min(X_[:, 0])
    xmax = max(X_[:, 0])
    ymin = min(X_[:, 1])
    ymax = max(X_[:, 1])
    indices = rng.choice(len(X_), size=(len(X_) * p) // 100, replace=False)
    projection_rnd = X_.copy()
    replacement = np.random.rand(len(indices), 2)
    replacement[:, 0] = xmin + replacement[:, 0] * (xmax - xmin)
    replacement[:, 1] = ymin + replacement[:, 1] * (ymax - ymin)
    projection_rnd[indices] = replacement
    return projection_rnd


def randomize_projections(X_):
    projections = [X_.copy()]
    for i in [5, 10, 25, 50, 100]:
        projections.append(randomize_projection(X_, i))
    return projections


#########################################################################
#########################################################################

# For reproducability of the results
np.random.seed(42)

sample_size = 1000

mnist = fetch_openml('mnist_784')

t_size = sample_size / mnist.data.shape[0]

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
X = tsne_results
# X = PCA(n_components=2).fit_transform(X)

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

projections = randomize_projections(X)

results_tw = []
results_s = []
f = lambda x: 1 / (1 + x)
for proj in projections:
    # Compute TW
    results_tw.append(2 * trustworthiness(X, proj) - 1)

    # Compute Sortedness
    s = []
    for coefname in [spearmanr, kendalltau, lambda X, X_: weightedtau(X, X_, weigher=f)]:
        s.append(sortedness(X, proj, f=coefname))
    s.append(sortedness(X, proj, f=None, weigher=f))
    results_s.append(s)

tw_mean = []
for tw in results_tw:
    tw_mean.append(mean(tw))

s_p_mean = []
s_tau_mean = []
s_wtau_x_mean = []
sortedness_mean = []
box_p = []
box_tau = []
box_wtau_x = []
box_sortedness_ = []
for s in results_s:
    s_p_mean.append(mean(s[0]))
    s_tau_mean.append(mean(s[1]))
    s_wtau_x_mean.append(mean(s[2]))
    sortedness_mean.append(mean(s[3]))
    box_p.append(s[0])
    box_tau.append(s[1])
    box_wtau_x.append(s[2])
    box_sortedness_.append(s[3])

df = pd.DataFrame({
    'Sortedness ρ': s_p_mean,
    'Sortedness tau': s_tau_mean,
    'Sortedness wtau-x': s_wtau_x_mean,
    'sortedness': sortedness_mean,
    'TW': tw_mean
})
ax = df.plot(kind='line')
ax.set_xticklabels(['', '0%', '5%', '10%', '25%', '50%', '100%'])
# ax.set_xticklabels([1,2,3,4,5,6], ['0%', '5%', '10%', '25%', '50%', '100%'])
plt.show()

#########################################
#########################################

fig = plt.figure()
bp = plt.boxplot(box_p + box_tau + box_wtau_x + box_sortedness_ + results_tw, vert=1, patch_artist=True)

bp_colors = np.repeat(list(map(plt.cm.Pastel1, ap[0, 1, ..., 4].l)), [6] * 5, axis=0)
bp_list = []
for i, bplot in enumerate(bp['boxes']):
    bplot.set(color='gray', linewidth=3)
    bplot.set(facecolor=bp_colors[i])
    bp_list.append(bplot)

for i, whisker in enumerate(bp['whiskers']):
    whisker.set(color='gray', linewidth=3)

for i, cap in enumerate(bp['caps']):
    cap.set(color='gray', linewidth=3)

for i, median in enumerate(bp['medians']):
    median.set(color='gray', linewidth=3)

plt.legend(
    list(map(bp_list.__getitem__, ap[0, 6, ..., 24])),
    ['Sortedness - p', 'Sortedness - tau', 'Sortedness - wtau-x', 'sortedness', 'TW'],
    loc='lower right'
)
plt.xticks(ap[1, 2, ..., 30], ['0%', '5%', '10%', '25%', '50%', '100%'] * 5)

plt.title("Boxplot", loc="center", fontsize=18)
plt.xlabel("Quantidade de pontos aleatorizados")
plt.ylabel("")
plt.show()
