import numpy as np


class threeCoin:
    def __init__(self, pi_, p, q):
        self.pi_ = pi_
        self.p = p
        self.q = q

    def e_step(self, y):
        self.mu = np.zeros_like(y)
        for i in range(y.shape[0]):
            self.mu[i] = self.pi_ * \
                np.power(self.p, y[i]) * np.power(1 - self.p, 1 - y[i])
            self.mu[i] = self.mu[i] / (self.mu[i] + (1 - self.pi_) *
                                       np.power(self.q, y[i]) * np.power(1 - self.q, 1 - y[i]))

    def m_step(self, y):
        self.pi_ = np.sum(self.mu) / self.mu.shape[0]
        self.p = np.sum(self.mu * y) / np.sum(self.mu)
        self.q = np.sum((1 - self.mu) * y) / np.sum(1 - self.mu)

    def predict(self, y, iter_n):
        for i in range(iter_n):
            self.e_step(y)
            self.m_step(y)
        
        return self.pi_, self.p, self.q

class TripleCoin(object):
    def __init__(self, pi=0, p=0, q=0):
        self.pi = pi
        self.p = p
        self.q = q

    def do_e_step(self, y):
        self.mu = np.zeros_like(y)
        for j, yj in enumerate(y):
            if yj == 1:
                self.mu[j] = self.pi * self.p / (self.pi * self.p + (1-self.pi) * self.q)
            else:
                self.mu[j] = self.pi * (1-self.p) / (self.pi * (1-self.p) + (1-self.pi) * (1-self.q))

    def do_m_step(self, y):
        self.pi_ = np.mean(self.mu, dtype=np.float)
        self.p = np.sum(self.mu * y) / np.sum(self.mu)
        self.q = np.sum((1 - self.mu) * y) / np.sum(1 - self.mu)

    def predict(self, y, iter_n):
        for i in range(iter_n):
            self.do_e_step(y)
            self.do_m_step(y)

        return self.pi_, self.p, self.q

if __name__ == '__main__':
    y = np.array([1., 1., 0., 1., 0., 0., 1., 0., 1., 1.])
    pi_, p, q = 0.4, 0.6, 0.7
    tc = threeCoin(pi_, p, q)
    pi_, p, q = tc.predict(y, 10)
    print(pi_, p, q)