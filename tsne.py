"""TSNE plotter

This script plots a t-SNE visualization for the data passed in
    * run - plots the t-SNE on the passed in matrix X and vector y
"""

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def run(X, y):
    """Processes stock data for neural network consumption
    Parameters
    ----------
    X : DataFrame
        DataFrame of features
    y : Series
        Classification data corresponding to X
    """
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(X)

    # scatter plot the sample points
    markers = ('x', 'x')
    color_map = {0: 'green', 1: 'red'}
    plt.figure()

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x_2d[y == cl, 0], y=x_2d[y == cl, 1],
                    c=color_map[idx], marker=markers[idx], label=cl)

    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization of test data')
    plt.show()
