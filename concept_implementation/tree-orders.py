# python3

import sys, threading
sys.setrecursionlimit(10**6) # max depth of recursion
threading.stack_size(2**25)  # new thread will get stack of such size
from pdb import set_trace as bkp

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class TreeOrders:
  def read(self):
    self.n = int(sys.stdin.readline())
    self.key = [0 for i in range(self.n)]
    self.left = [0 for i in range(self.n)]
    self.right = [0 for i in range(self.n)]
    for i in range(self.n):
      [a, b, c] = map(int, sys.stdin.readline().split())
      self.key[i] = a #value
      self.left[i] = b #index
      self.right[i] = c #index
    self.root = self.create_bst(0)

  def create_bst(self,root_index):
    # stop condition

    root = TreeNode(self.key[root_index])
    if self.left[root_index] != -1: 
        root.left = self.create_bst(self.left[root_index])

    if self.right[root_index] != -1:
        root.right = self.create_bst(self.right[root_index])
        
    return root
    

  def inOrder(self):
    self.result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    root = self.root
    #iterative method
    stack = []
    while root or stack:
        if root:
            stack.append(root)
            root = root.left
        else:
            top = stack.pop()
            self.result.append(top.val)
            root = top.right
    return self.result

  def preOrder(self):
    self.result = []
    root = self.root
    # Finish the implementation
    # You may need to add a new recursive method to do that
    #iterative method
    stack = []
    while root or stack:
        if root:
            stack.append(root)
            self.result.append(root.val)
            root = root.left
        else:
            top = stack.pop()
            root = top.right
    return self.result

  def postOrder(self):
    self.result = []
    root = self.root
    # Finish the implementation
    # You may need to add a new recursive method to do that
    stack = []
    while root or stack:
        if root:
            stack.append(root)
            self.result.append(root.val)
            root = root.right
        else:
            top = stack.pop()
            root = top.left
    return self.result[::-1]

def main():
	tree = TreeOrders()
	tree.read()
	print(" ".join(str(x) for x in tree.inOrder()))
	print(" ".join(str(x) for x in tree.preOrder()))
	print(" ".join(str(x) for x in tree.postOrder()))

threading.Thread(target=main).start()
