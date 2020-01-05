# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/5 17:29   guzhouweihu      1.0         None
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from knn import KNN

if __name__ == '__main__':
    # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length',
        'sepal width',
        'petal length',
        'petal width',
        'label']
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    data = np.array(df.iloc[:, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = KNN(X_train, y_train)
    print(clf.score(X_test, y_test))

    test_point = [6.0, 3.0]
    print('Test Point: {}'.format(clf.predict(test_point)))

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'ro', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


