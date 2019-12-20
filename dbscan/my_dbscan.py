import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import matplotlib.pyplot as plt

class DBSCANMy:
    def __init__(self, Datas, eps, minPts):
        '''

        :param Datas: 需要聚类的数据
        :param eps:
        :param minPts:
        '''
        self.__UNVISIT = -2     # 标记，表示该点是否被遍历过
        self.__NOISE = -1       # 标记，记为该点为噪声点

        self.C = -1             # 当前的类
        self.Datas = Datas
        self.eps = eps
        self.minPts = minPts
        self.dist_matrix = np.full((Datas.shape[0], Datas.shape[0]), float(self.__UNVISIT))     # 记录点到点距离的矩阵
        self.target = np.full((Datas.shape[0]), self.__UNVISIT)                              # 记录目标点归属类的数组


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
                if len(queryResults) >= self.minPts:
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
        self.C += 1
        nPoints = self.Datas.shape[0]
        for pointId in range(nPoints):
            if self.target[pointId] == self.__UNVISIT:
                if self.expand_cluster(pointId):
                    self.C += 1

        return self.target

    def DBSCAN(self, show_plot=False, show_data=None, update_iter=50):
        iter_num = 0
        if show_plot:
            plt.cla()
            plt.scatter(show_data[:, 0], show_data[:, 1], c=self.target, marker='.')
            plt.pause(0.001)
            # plt.title('begin_plot:')
            # plt.show()

        for P in range(self.Datas.shape[0]):
            # 检查当前点是否被遍历过
            if self.target[P] != self.__UNVISIT:
                continue
            Neighbors1 = self.region_query(P)
            if len(Neighbors1) < self.minPts:
                self.target[P] = self.__NOISE
                continue

            # 开始一个新的聚类过程，因此聚类号加一
            self.C += 1
            self.target[P] = self.C
            # 查看以当前点为中心，eps为半径的范围内存在多少个数据点
            seeds = np.setdiff1d(Neighbors1, P).tolist()
            while len(seeds) > 0:
                # 画出中间图
                
                # 获取下一个点
                Q = seeds[0]
                if self.target[Q] == self.__NOISE:
                    self.target[Q] = self.C
                if self.target[Q] != self.__UNVISIT:
                    seeds = seeds[1:]
                    continue
                self.target[Q] = self.C
                Neighbors2 = self.region_query(Q)
                if len(Neighbors2) >= self.minPts:
                    for i in range(len(Neighbors2)):
                        if Neighbors2[i] not in seeds:
                            seeds.append(Neighbors2[i])
                seeds = seeds[1:]
                iter_num += 1
                if show_plot and (iter_num % update_iter == 0):
                    plt.cla()
                    plt.scatter(show_data[:, 0], show_data[:, 1], c=self.target, marker='.')
                    plt.pause(0.001)
                    # plt.title(f'iter_plot, num: {iter_num}')
                    # plt.show()

        plt.cla()
        plt.scatter(show_data[:, 0], show_data[:, 1], c=self.target, marker='.')
        return self.target

    def region_query(self, Q):
        '''

        :param Q: 点Q为当前点，需要计算该点附件有多少点距离该点的距离小于eps
        :return:
        '''
        Neighbors = []
        for P in range(self.Datas.shape[0]):
            if self.__distFunc(Q, P) < self.eps:
                if P == Q:
                    continue
                Neighbors.append(P)
        return np.array(Neighbors)

    def __distFunc(self, Q, P):
        '''

        :param Q: 点Q为当前点，需要计算该点附件有多少点距离该点的距离小于eps
        :param P: 与Q点不同的点
        :return:
        '''
        if self.dist_matrix[Q, P] != self.__UNVISIT:
            return self.dist_matrix[Q, P]
        else:
            # dist = np.sqrt(np.power(self.Datas[Q] - self.Datas[P], 2).sum())
            dist = np.linalg.norm(self.Datas[Q] - self.Datas[P])
            self.dist_matrix[Q, P] = self.dist_matrix[P, Q] = dist
            return dist

def get_data3():
    eps = .3
    minPnts = 10

    noisy_moons = make_moons(n_samples=1500, noise=.05)[0]
    return eps, minPnts, noisy_moons


def get_data2():
    eps = 0.1
    minPnts = 10
    X1, Y1 = make_circles(n_samples=1500, factor=0.6, noise=0.05,
                          random_state=1)
    X2, Y2 = make_blobs(n_samples=300, n_features=2, centers=[[1.5, 1.5]],
                        cluster_std=[[0.1]], random_state=5)

    x = np.concatenate((X1, X2))
    return eps, minPnts, x

def get_data1():
    eps = 0.3
    minPnts = 5
    # x, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                             random_state=0)
    random_state = 170
    X, y = make_blobs(n_samples=1500, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    return eps, minPnts, aniso[0]



if __name__ == '__main__':
    # #############################################################################
    # Generate sample data

    np.random.seed(0)


    update_iter_x = input('每分类x个点更新图片，x=')
    use_default = input('使用默认数据集((y)/n)：')
    input_data = ''
    eps, minPnts, x = None, None, None

    if use_default == '' or use_default == 'y' or use_default == 'Y':
        print('默认数据集：（1，三个三点聚类集--通过sklearn中生成blobs）')
        print('默认数据集：（2，两环一团聚类集--通过sklearn中生成blobs）')
        print('默认数据集：（3，两个月牙聚类集--通过sklearn中生成blobs）')

        data_select = input('请选择: ')
        data_switcher = {
            '1': get_data1,
            '2': get_data2,
            '3': get_data3
        }


        eps, minPnts, input_data = data_switcher[data_select]()


    else:
        file_path = input('请输入数据文件路径：')
        input_data = np.loadtxt(file_path)
        eps = input('eps=')
        eps = float(eps)
        minPnts = input('minPnts=')
        minPnts = int(minPnts)


    dbs = DBSCANMy(input_data, eps, minPnts)

    plt.ion()
    target = dbs.DBSCAN(show_plot=True, show_data=input_data, update_iter=int(update_iter_x))

    plt.ioff()
    plt.cla()
    
    plt.scatter(x=input_data[:, 0], y=input_data[:, 1], c=target, marker='.')
    plt.show()