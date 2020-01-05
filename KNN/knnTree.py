# -*- encoding: utf-8 -*-
'''
@File    :   knnTree.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/5 18:10   guzhouweihu      1.0         None
'''

import numpy as np
from KNN.kdTree import kdTree

class KNN(object):
    def __init__(self, k_neighbors=3, p=2):
        self.X_train = None
        self.Y_train = None
        self.k = k_neighbors
        self.p = p
        self.kdt = None

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        self.kdt = kdTree(self.X_train)

    def predict_list(self, X):
        if self.X_train is None or self.Y_train is None:
            print("Model has not been trained before prediction")
            return None


        predict_X = np.full(X.shape[0], -1)

        class_set = set(self.Y_train)
        for i in X.shape[0]:
            pred_res = self.kdt.nearest_useMyqueue(X[i], self.k, self.p)

            pred_counter = {}

            for each_class in class_set:
                pred_counter[each_class] = 0
            for each_pred in pred_res:
                pred_counter[self.Y_train[self.X_train.index(each_pred)]] += 1

            predict_X[i] = max(pred_counter, key=pred_counter.get)

        return predict_X

    def predict_one(self, target):
        if target is None:
            print("Model has not been trained before prediction")
            return None

        nearest_point = self.kdt.nearest_useMyqueue(target, self.k, self.p)

        pred_counter = {}
        class_set = set(self.Y_train)
        for each_class in class_set:
            pred_counter[each_class] = 0
        for each_pred in nearest_point:
            pred_counter[self.Y_train[self.X_train.index(each_pred)]] += 1

        return max(pred_counter, key=pred_counter.get), nearest_point


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt


    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length',
        'sepal width',
        'petal length',
        'petal width',
        'label']
    # plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    # plt.xlabel('sepal length')
    # plt.ylabel('sepal width')
    # plt.legend()
    # plt.show()


    data = np.array(df.iloc[:, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNN(k_neighbors=5)
    knn.fit(X_train.tolist(), y_train.tolist())
    test_point = [6., 2.75]
    pred, nearest_point = knn.predict_one(test_point)

    print('Test Point: {}'.format(pred))

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'go', label='test_point')
    for near in nearest_point:
        plt.plot(near[0], near[1], 'ro', label='nearest_point')
        print(near)

    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()