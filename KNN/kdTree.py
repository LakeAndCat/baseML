# -*- encoding: utf-8 -*-
'''
@File    :   kdTree.py
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/4 21:27   guzhouweihu      1.0         None
'''

import math
from collections import namedtuple
import time
from random import random
import numpy as np
import heapq
from priority_queue import MaxHeap

class kdNode(object):
    def __init__(self, point, split_dim, left, right):
        self.point = point
        self.split_dim = split_dim
        self.left = left
        self.right = right


class kdTree(object):
    def __init__(self, dataset):
        self.k = len(dataset[0])

        def createNode(split_dim, dataset):
            if not dataset:
                return None

            dataset.sort(key=lambda x: x[split_dim])
            split_pos = len(dataset) // 2
            mid_pnt = dataset[split_pos]
            split_next_dim = (split_dim + 1) % self.k

            return kdNode(mid_pnt, split_dim,
                          createNode(split_next_dim, dataset[:split_pos]),
                          createNode(split_next_dim, dataset[split_pos + 1:]))

        self.root = createNode(0, dataset)

    def nearest(self, x, near_k=1, p=2):
        self.k_nearest = [(-np.inf, None)] * near_k
        def travel(node):
            if not node == None:
                dist = x[node.split_dim] - node.point[node.split_dim]

                travel(node.left if dist < 0 else node.right)

                cur_dist = np.linalg.norm(np.array(x) - np.array(node.point), p)

                heapq.heappushpop(self.k_nearest, (-cur_dist, node))

                if -(self.k_nearest[0][0]) > abs(dist):
                    travel(node.right if dist < 0 else node.left)


        travel(self.root)
        self.k_nearest = np.array([i[1].point for i in heapq.nlargest(near_k, self.k_nearest)])
        return self.k_nearest

    def nearest_useMyqueue(self, target, near_k=1, p=2):
        priorityQueue = MaxHeap(near_k, lambda x: x[0])
        def travel(node):
            if not node == None:
                dist = target[node.split_dim] - node.point[node.split_dim]
                travel(node.left if dist < 0 else node.right)

                cur_dist = np.linalg.norm(np.array(target) - np.array(node.point), p)

                priorityQueue.insert_keep_size((cur_dist, node))
                max_item = priorityQueue.get_max()
                if max_item[0] > abs(dist):
                    travel(node.right if dist < 0 else node.left)

        travel(self.root)
        res = []
        for i in range(near_k):
            res.insert(0, priorityQueue.del_max()[1].point)
        return res


# 中序遍历
def preorder(root):
    if root.left:
        preorder(root.left)
    print(root.point)
    if root.right:
        preorder(root.right)


if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = kdTree(data)
    preorder(kd.root)

    # 产生一个k维随机向量，每维分量值在0~1之间
    def random_point(k):
        return [random() for _ in range(k)]

    # 产生n个k维随机向量

    def random_points(k, n):
        return [random_point(k) for _ in range(n)]

    N = 40000
    data = random_points(3, N)
    kd2 = kdTree(data)            # 构建包含四十万个3维空间样本点的kd树


    result2 = kd2.nearest([0.1, 0.5, 0.8], 5, 2)
    result3 = kd2.nearest_useMyqueue([0.1, 0.5, 0.8], 5, 2)
    print(result2)
    print(np.array(result3))



