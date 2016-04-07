"""
mehod1
python heapq https://docs.python.org/2/library/heapq.html
"""

"""
#http://bookshadow.com/weblog/2015/10/19/leetcode-find-median-data-stream/
minHeap = Heap(cmp = lambda x, y, a: a[x] < a[y])
maxHeap = Heap(cmp = lambda x, y, a: a[x] > a[y])
"""
class Heap:
    def __init__(self, cmp):
        self.cmp = cmp
        self.heap = [None]

    def __swap__(self, x, y, a):
        a[x], a[y] = a[y], a[x]

    def size(self):
        return len(self.heap) - 1

    def top(self):
        return self.heap[1] if self.size() else None

    def append(self, num):
        self.heap.append(num)
        self.siftUp(self.size())

    def pop(self):
        top, last = self.heap[1], self.heap.pop()
        if self.size():
            self.heap[1] = last
            self.siftDown(1)
        return top

    def siftUp(self, idx):
        while idx > 1 and self.cmp(idx, idx / 2, self.heap):
            self.__swap__(idx / 2, idx, self.heap)
            idx /= 2

    def siftDown(self, idx):
        while idx * 2 <= self.size():
            nidx = idx * 2
            #comapare left child and right child
            # since help]0] is None, so left child is no longer i*2+1 but i*2
            # right child is no longer i*2+2 but i*2+1
            if nidx + 1 <= self.size() and self.cmp(nidx + 1, nidx, self.heap):
                nidx += 1
            if self.cmp(nidx, idx, self.heap):
                self.__swap__(nidx, idx, self.heap)
                idx = nidx
            else:
                break


#http://bookshadow.com/weblog/2015/08/14/leetcode-the-skyline-problem/
class MaxHeap:
    def __init__(self, buildings):
        self.buildings = buildings
        self.size = 0
        self.heap = [None] * (2 * len(buildings) + 1)
        self.lineMap = dict()
    def maxLine(self):
        return self.heap[1]
    def insert(self, lineId):
        self.size += 1
        self.heap[self.size] = lineId
        self.lineMap[lineId] = self.size
        self.siftUp(self.size)
    def delete(self, lineId):
        heapIdx = self.lineMap[lineId]
        self.heap[heapIdx] = self.heap[self.size]
        self.lineMap[self.heap[self.size]] = heapIdx
        self.heap[self.size] = None
        del self.lineMap[lineId]
        self.size -= 1
        self.siftDown(heapIdx)
    def siftUp(self, idx):
        while idx > 1 and self.cmp(idx / 2, idx) < 0:
            self.swap(idx / 2, idx)
            idx /= 2
    def siftDown(self, idx):
        while idx * 2 <= self.size:
            nidx = idx * 2
            if idx * 2 + 1 <= self.size and self.cmp(idx * 2 + 1, idx * 2) > 0:
                nidx = idx * 2 + 1
            if self.cmp(nidx, idx) > 0:
                self.swap(nidx, idx)
                idx = nidx
            else:
                break
    def swap(self, a, b):
        la, lb = self.heap[a], self.heap[b]
        self.lineMap[la], self.lineMap[lb] = self.lineMap[lb], self.lineMap[la]
        self.heap[a], self.heap[b] = lb, la
    def cmp(self, a, b):
        return self.buildings[self.heap[a]][2] - self.buildings[self.heap[b]][2]





"""
特点

以数组表示，但是以完全二叉树的方式理解。
唯一能够同时最优地利用空间和时间的方法——最坏情况下也能保证使用 2N \log N2NlogN 次比较和恒定的额外空间。
在索引从0开始的数组中：
父节点 i 的左子节点在位置(2*i+1)
父节点 i 的右子节点在位置(2*i+2)
子节点 i 的父节点在位置floor((i-1)/2)

"""
class MaxHeap:
    def __init__(self, array=None):
        if array:
            self.heap = self._max_heapify(array)
        else:
            self.heap = []

    def _sink(self, array, i):
        # move node down the tree
        left, right = 2 * i + 1, 2 * i + 2
        max_index = i
        if left < len(array) and array[left] > array[max_index]:
            max_index = left
        if right < len(array) and array[right] > array[max_index]:
            max_index = right
        if max_index != i:
            array[i], array[max_index] = array[max_index], array[i]
            self._sink(array, max_index)

    def _swim(self, array, i):
        # move node up the tree
        if i == 0:
            return
        father = (i - 1) / 2
        if array[father] < array[i]:
            array[father], array[i] = array[i], array[father]
            self._swim(array, father)

    def _max_heapify(self, array):
        for i in xrange(len(array) / 2, -1, -1):
            self._sink(array, i)
        return array

    def push(self, item):
        self.heap.append(item)
        self._swim(self.heap, len(self.heap) - 1)

    def pop(self):
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        item = self.heap.pop()
        self._sink(self.heap, 0)
        return item