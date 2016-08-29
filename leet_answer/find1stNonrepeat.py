#Finding first non repeating character in a string array

class Node(object):
    def __init__(self,key=None,count = 0):
        self.key = key
        self.count = count
        self.next = None


class solution():
    def find1stNonRepeat(self,input):
        dummy = Node()
        tail = dummy
        tb = {} #key:node
        for x in input:
            if x not in tb:
                nd = Node(x,1)
                tail.next = nd
                tail = tail.next
                tb[x] = nd
            else:
                tb[x].count = tb[x].count +1
        curr = dummy.next
        while curr:
            if curr.count ==1:
                return curr.key
            curr = curr.next
        return None



rawinput = "geeksforgeeks"
sol = solution()
c = sol.find1stNonRepeat(rawinput)
print c