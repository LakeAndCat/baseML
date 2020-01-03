# -*- encoding: utf-8 -*-
'''
@File    :   LDA_multi_class.py
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/3 15:21   guzhouweihu      1.0         None
'''

import numpy as np
import matplotlib.pyplot as plt


def read_iris():
    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    data_set = load_iris()
    data_x = data_set.data
    label = data_set.target + 1
    #preprocessing.scale(data_x, axis=0, with_mean=True, with_std=True, copy=False)
    return data_x, label


def class_mean(data, label, clusters):
    mean_vec = []
    for c in range(1, clusters + 1):
        mean_vec.append(np.mean(data[label == c, :], axis=0))

    return np.array(mean_vec)


def with_class_SW(data, label, clusters):
    m = data.shape[1]
    SW = np.zeros((m, m))
    mean_vec = class_mean(data, label, clusters)

    for c, mv in zip(range(1, clusters + 1), mean_vec):
        class_sc_mat = np.zeros((m, m))

        SW += (data[label == c] - mv).T @ (data[label == c] - mv)

    return SW


def between_class_SB(data, label, clusters):
    m = data.shape[1]
    all_mean = np.mean(data, axis=0)
    SB = np.zeros((m, m))
    mean_vec = class_mean(data, label, clusters)
    for c, mean_vec in enumerate(mean_vec):
        n = data[label == c + 1, :].shape[0]
        mean_vec = mean_vec[:, np.newaxis]
        all_mean = all_mean.reshape(-1, 1)
        SB += n * ((mean_vec - all_mean) @ (mean_vec - all_mean).T)

    return SB


def LDA(data, label, clusters):
    S_W = with_class_SW(data, label, clusters)
    S_B = between_class_SB(data, label, clusters)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i].reshape(4, 1)
        print('\nEigenvector {}: \n{}'.format(i + 1, eigvec_sc.real))
        print('Eigenvalue {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack((eig_pairs[0][1].reshape(-1, 1),
                   eig_pairs[1][1].reshape(-1, 1)))
    print('Matrix W:\n', W.real)
    print(data.dot(W))
    return W


def plot_lda():
    data, labels = read_iris()
    W = LDA(data, labels, 3)

    Y = data.dot(W)
    # print Y
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=Y[:, 0][labels == label],
                    y=Y[:, 1][labels == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    )
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.show()


def default_plot_lda():
    Y = sklearnLDA()
    data, labels = read_iris()
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=Y[:, 0][labels == label],
                    y=Y[:, 1][labels == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    )
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.title('LDA:default')

    plt.show()


def sklearnLDA():
    from sklearn import datasets
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    lda = LDA(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    return X_r2


if __name__ == "__main__":
    # lda()
    sklearnLDA()
    plot_lda()
    default_plot_lda()
