# -*- encoding: utf-8 -*-
'''
@File    :   gmm.py
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2019/12/28 20:54   guzhouweihu      1.0         None
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import mixture
import time


class gmm:
    def __init__(self, shape, K):
        self.N, self.D = shape
        self.K = K

        self.mu = np.random.rand(self.K, self.D)            # K * D
        self.cov = np.array([np.eye(self.D)] * self.K)      # K * D * D
        self.alpha = np.array([1.0 / self.K] * self.K)      # 1 * K

    def _prob(self, X, k):
        phi = multivariate_normal(mean=self.mu[k], cov=self.cov[k])
        return phi.pdf(X)

    def _phi(self, X):
        phi = np.zeros((self.N, self.K))
        coef1 = np.power((2 * np.pi), self.D / 2.0)
        for k in range(self.K):
            # 1 by 1, la.det计算行列式
            coef2 = np.power(la.det(self.cov[k]), 0.5)
            coef3 = 1 / (coef1 * coef2)
            shift = X - self.mu[k]
            sigma = la.inv(self.cov[k])
            e_part = np.exp(np.sum(-0.5 * shift.dot(sigma) * shift, axis=1))

            phi[:, k] = coef3 * e_part

        return phi

    def _e_step(self, X):

        # 响应度矩阵
        gamma = np.mat(np.zeros((self.N, self.K)))
        prob = np.zeros((self.N, self.K))

        for k in range(self.K):
            prob[:, k] = self._prob(X, k)
        prob = np.mat(prob)

        for k in range(self.K):
            gamma[:, k] = self.alpha[k] * prob[:, k]

        for i in range(self.N):
            gamma[i, :] /= np.sum(gamma[i, :])


        return gamma



    def _m_step(self, X, gamma):

        for k in range(self.K):
            Nk = np.sum(gamma[:, k])

            self.mu[k, :] = np.sum(np.multiply(gamma[:, k], X), axis=0) / Nk

            self.cov[k, :, :] = (X - self.mu[k]).T * \
                np.multiply((X - self.mu[k]), gamma[:, k]) / Nk

            self.alpha[k] = Nk / self.N


    def _e_step_by_matrix(self, X):

        prob = self._phi(X)

        gamma = np.multiply(prob, self.alpha)

        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

        return gamma

    def _m_step_by_matrix(self, X, gamma):

        NK = np.sum(gamma, axis=0)  # 1 * K

        # (K,N) @ (N,D) / (K,) = (K,D)
        self.mu = gamma.T @ X / NK  # K * D


        XK = np.array([X] * self.K)  # K * N * D
        # x_mu = (XK.transpose(1, 0, 2) - self.mu).transpose(1, 0, 2)
        x_mu = XK - self.mu[:, np.newaxis, :]  # K * N * D

        # (K,D,N) @ ((K,N,D) * (K,N,1)) / (K,1,1) = (K,D,D)
        self.cov = x_mu.transpose(0, 2, 1) @ (np.multiply(
            x_mu, gamma.T[:, :, np.newaxis])) / NK[:, np.newaxis, np.newaxis]       
        self.alpha = NK / self.N

    def fit(self, X, threshold):
        X = self._scale_data(X)
        preL = -np.inf
        while True:
            gamma = self._e_step_by_matrix(X)
            self._m_step_by_matrix(X, gamma)
            curL = self._likehoodloss(X)

            if self._stop_iterator_strategy(threshold, curL, preL):
                break

            preL = curL

        gamma = self._e_step(X)
        cluster = np.argmax(gamma, axis=1)

        return cluster

    def _scale_data(self, X):
        for i in range(X.shape[1]):
            max_ = X[:, i].max()
            min_ = X[:, i].min()
            X[:, i] = (X[:, i] - min_) / (max_ - min_)

        return X

    def _likehoodloss(self, X):
        curL = np.sum(np.log(np.sum(X * self.alpha, axis=1)))
        return curL

    def _stop_iterator_strategy(self, threshold, predL, curL):
        return np.abs(curL - predL) < threshold


def scale_data(X):
    for i in range(X.shape[1]):
        max_ = X[:, i].max()
        min_ = X[:, i].min()
        X[:, i] = (X[:, i] - min_) / (max_ - min_)

    return X

if __name__ == '__main__':
    # ------data1--------
    # X = np.loadtxt('gmm.data')

    # ------data2--------
    n_samples = 500
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


    model = gmm(X.shape, 2)

    start1 = time.time()
    cluster = model.fit(X.copy(), 1e-15)
    cluster = np.array(cluster).squeeze()
    end1 = time.time()
    print(end1 - start1)      # data2--使用矩阵运算与for循环时间差大约12倍

    plt.scatter(X[:, 0], X[:, 1], c=cluster, marker='.')
    plt.show()


    start2 = time.time()

    clf = mixture.GaussianMixture(
        n_components=2,
        covariance_type='diag').fit(scale_data(X))      # 同样条件下，sklearn库中的方法快10倍

    end2 = time.time()
    print(end2 - start2)

    plt.scatter(X[:, 0], X[:, 1], c=clf.predict(X), marker='.')
    plt.show()
