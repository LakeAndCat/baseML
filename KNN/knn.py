# -*- encoding: utf-8 -*-
'''
@File    :   knn.py
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/3 20:35   guzhouweihu      1.0         None
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter, namedtuple
from sklearn.neighbors import kneighbors_graph



class KNN:
    def __init__(self, X_train, y_train, k_neighbors=3, p=2):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k_neighbors
        self.p = 2

    def predict(self, X):
        # 取出k个点进行预测
        knn_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X - self.X_train, ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.k, len(self.X_train)):
            max_dist_idx = knn_list.index(max(knn_list, key=lambda x: x[0]))
            now_dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if now_dist < knn_list[max_dist_idx][0]:
                knn_list[max_dist_idx] = (now_dist, self.y_train[i])

        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            if self.predict(X) == y:
                right_count += 1
            # right_count += 1 if self.predict(X) == y else 0
        return right_count / len(y_test)




