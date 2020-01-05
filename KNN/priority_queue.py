# -*- encoding: utf-8 -*-
'''
@File    :   priority_queue.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/1/5 14:51   guzhouweihu      1.0         None
'''


class MaxHeap(object):
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self._items = [None] * max_size
        self.N = 0
        pass

    def __str__(self):
        item_values = str([self.fn(x) for x in self._items])
        info = (self.N, self.max_size, self._items, item_values)
        return "Size: %d\nMax size: %d\nItems: %s\nItem_values: %s\n" % info

    @property
    def items(self):
        return self._items[: self.N]

    @property
    def is_full(self):
        return self.N == self.max_size

    def _exch(self, i, j):
        self._items[i], self._items[j] = self._items[j], self._items[i]

    def _less(self, i, j):
        return self.fn(self._items[i]) < self.fn(self._items[j])

    def _value(self, idx):
        item = self._items[idx]
        if item is None:
            return -float('inf')
        else:
            return self.fn(item)

    def _swim(self, k):
        while k > 0 and self._less((k - 1)//2, k):
            self._exch((k - 1)//2, k)
            k = (k - 1)//2

    def _sink(self, k):
        while 2*(k+1)-1 < self.N:
            j = 2*(k+1)-1
            if j+1 < self.N and self._less(j, j+1):
                j += 1
            if not self._less(k, j):
                break
            self._exch(k, j)
            k = j

    def insert(self, item):
        self._items[self.N] = item
        self.N += 1
        self._swim(self.N-1)

    def insert_keep_size(self, item):

        if self.is_full:
            if self.fn(item) < self.fn(self.items[0]):
                self._items[0] = item
                self._sink(0)
        else:
            # assert self.N == 3
            self.insert(item)

    def del_max(self):
        result = self._items[0]
        self._exch(0, self.N-1)
        self.N -= 1
        self._items[self.N] = None
        self._swim(self.N-1)

        return result

    def get_max(self):
        return self._items[0]

