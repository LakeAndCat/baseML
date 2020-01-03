# -*- encoding: utf-8 -*-
'''
@File    :   LDA.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/3 13:23   guzhouweihu      1.0         None
'''

import numpy as np
import matplotlib.pyplot as plt

def compute_st(x_i):
    mu_i = np.mean(x_i, axis=0)
    return (x_i - mu_i).T @ (x_i - mu_i)

def between_class_SB(X1, X2):

    mu_1 = np.mean(X1, axis=0)[np.newaxis, :]
    mu_2 = np.mean(X2, axis=0)[np.newaxis, :]
    # mu = np.mean(np.row_stack((X1, X2)), axis=0)[np.newaxis, :]

    # return (mu - mu_1).T @ (mu - mu_1) + (mu - mu_2).T @ (mu - mu_2)
    return (mu_1-mu_2).T @ (mu_1-mu_2)

def LDA(X1, X2):
    # 计算类内离散度矩阵Sw
    s1 = compute_st(X1)
    s2 = compute_st(X2)

    Sw = s1 + s2
    Sb = between_class_SB(X1, X2)
    eig_value, eig_vec = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    index_vec = np.argsort(-eig_value)
    w = eig_vec[:, index_vec[:1]]
    return w

def LDA1(X1, X2):
    # 计算类内离散度矩阵Sw
    s1 = compute_st(X1)
    s2 = compute_st(X2)

    Sw = s1 + s2
    mu_1 = np.mean(X1, axis=0)[np.newaxis, :]
    mu_2 = np.mean(X2, axis=0)[np.newaxis, :]

    return ((mu_1 - mu_2) @ np.linalg.inv(Sw)).T


#构造数据库
def createDataSet():
    X1 = np.random.random((8, 2)) * 5 + 15  #类别A
    X2 = np.random.random((8, 2)) * 5 + 2   #类别B
    return X1, X2

if __name__ == "__main__":
    x1, x2 = createDataSet()
    w = LDA(x1, x2)
    w1 = LDA1(x1, x2)

    plt.scatter(x1[:, 0], x1[:, 1], c='red', marker='.')
    plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='.')
    plt.show()

    plt.scatter((x1@w1).squeeze(), [1] * x1.shape[0], c='red', marker='.')
    plt.scatter((x2@w1).squeeze(), [2] * x2.shape[0], c='blue', marker='.')
    plt.show()
