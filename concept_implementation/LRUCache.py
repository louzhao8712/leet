class Node(object):
    def __init__(self,key,value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class doubleLinkList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def removeLast(self):
        self.remove(self.tail)

    def remove(self,node):
        if self.head == self.tail:
            self.head,self.tail = None,None
            return
        if node == self.head:
            self.head.next.prev = None
            self.head = node.next
            return
        if node == self.tail:
            node.prev.next = None
            self.tail = node.prev
            return
        node.prev.next = node.next
        node.next.prev = node.prev

    def addFirst(self,node):
        if not self.head:
            self.head = self.tail = node
            node.prev = node.next = None
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            node.prev = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.tb = {} #key node
        self.cache = doubleLinkList()
        self.capacity = capacity
        self.size = 0


    def get(self, key):
        """
        :rtype: int
        """
        if key in self.tb:
            self.cache.remove(self.tb[key])
            self.cache.addFirst(self.tb[key])
            return self.tb[key].value
        else:
            return -1


    def set(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: nothing
        """
        if key in self.tb:
            self.cache.remove(self.tb[key])
            self.cache.addFirst(self.tb[key])
            self.tb[key].value = value
        else:
            node = Node(key,value)
            self.tb[key] = node
            self.size +=1

            if self.size > self.capacity:
                self.size -= 1
                del self.tb[self.cache.tail.key]
                self.cache.removeLast()
            self.cache.addFirst(node)
