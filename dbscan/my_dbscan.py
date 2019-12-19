import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import time

class DBSCANMy:
    def __init__(self, Datas, eps, minPts):
        '''

        :param Datas: 需要聚类的数据
        :param eps:
        :param minPts:
        '''
        self.__UNVISIT = -2     # 标记，表示该点是否被遍历过
        self.__NOISE = -1       # 标记，记为该点为噪声点

        self.C = 0              # 当前的类
        self.Datas = Datas
        self.eps = eps
        self.minPts = minPts
        self.dist_matrix = np.full((Datas.shape[0], Datas.shape[0]), float(self.__UNVISIT))     # 记录点到点距离的矩阵
        self.target = np.full((Datas.shape[0], 1), self.__UNVISIT)                              # 记录目标点归属类的数组


    def expand_cluster(self, pointId):
        
        seeds = self.region_query(pointId).tolist()
        if len(seeds) < self.minPts:
            self.target[pointId] = self.__NOISE
            return False
        else:
            self.target[pointId] = self.C
            for seedId in seeds:
                self.target[seedId] = self.C

            while len(seeds) > 0:
                currentPoint = seeds[0]
                queryResults = self.region_query(currentPoint)
                if len(queryResults) > self.minPts:
                    for i in range(len(queryResults)):
                        resultPoint = queryResults[i]
                        if self.target[resultPoint] == self.__UNVISIT:
                            seeds.append(resultPoint)
                            self.target[resultPoint] = self.C
                        elif self.target[resultPoint] == self.__NOISE:
                            self.target[resultPoint] = self.C
                seeds = seeds[1:]
        return True

    def dbscan_(self):
        nPoints = self.Datas.shape[0]
        for pointId in range(nPoints):
            if self.target[pointId] == self.__UNVISIT:
                if self.expand_cluster(pointId):
                    self.C += 1

        return self.target

    def DBSCAN(self):
        for P in range(self.Datas.shape[0]):
            if self.target[P, 0] != self.__UNVISIT:
                continue
            Neighbors1 = self.region_query(P)
            if len(Neighbors1) < self.minPts:
                self.target[P] = self.__NOISE
                continue
            self.C += 1
            self.target[P, 0] = self.C
            seeds = np.setdiff1d(Neighbors1, P).tolist()
            while len(seeds) > 0:
                Q = seeds[0]
                if self.target[Q, 0] == self.__NOISE:
                    self.target[Q, 0] = self.C
                if self.target[Q, 0] != self.__UNVISIT:
                    seeds = seeds[1:]
                    continue
                self.target[Q, 0] = self.C
                Neighbors2 = self.region_query(Q)
                if len(Neighbors2) >= self.minPts:
                    for i in range(len(Neighbors2)):
                        if Neighbors2[i] not in seeds:
                            seeds.append(Neighbors2[i])
                seeds = seeds[1:]


        return self.target

    def region_query(self, Q):
        Neighbors = []
        for P in range(self.Datas.shape[0]):
            if self.__distFunc(Q, P) < self.eps:
                Neighbors.append(P)
        return np.array(Neighbors)

    def __distFunc(self, Q, P):
        if self.dist_matrix[Q, P] != self.__UNVISIT:
            return self.dist_matrix[Q, P]
        else:
            # dist = np.sqrt(np.power(self.Datas[Q] - self.Datas[P], 2).sum())
            dist = np.linalg.norm(self.Datas[Q] - self.Datas[P])
            self.dist_matrix[Q, P] = self.dist_matrix[P, Q] = dist
            return dist


if __name__ == '__main__':
    # from sklearn import datasets
    # #
    # iris = datasets.load_iris()
    # iris_data = iris['data']
    # dbs = DBSCANMy(iris_data, 3, 2)
    # dbs_target = dbs.DBSCAN()
    #
    # # sklearn_target = sklearn.cluster.DBSCAN.

    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    start = time.time()

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    end1 = time.time()

    dbs = DBSCANMy(X, 0.3, 10)
    target = dbs.dbscan_().reshape((1, -1))

    target_D = dbs.DBSCAN().reshape((1, -1))


    end2 = time.time()

    print(end1 - start)

    print(end2 - end1)