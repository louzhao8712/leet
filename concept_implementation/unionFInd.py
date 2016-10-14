#261. Graph Valid Tree
"""
 Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
 write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.

Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.

Hint:

    Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], what should your return? Is this case a valid tree?
    According to the definition of tree on Wikipedia: 
    “a tree is an undirected graph in which any two vertices are connected by exactly one path. 
    In other words, any connected graph without simple cycles is a tree.”

"""
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        #detect cycle in undirected graph
        #Has n-1 edges and is acyclic.
        #Has n-1 edges and is connected.
        #https://discuss.leetcode.com/topic/21737/8-10-lines-union-find-dfs-and-bfs
        return self.sol4(n,edges)
        
    def sol1(self,n,edges):
        #union find solution
        parent = range(n)
        def find(x):
            return x if parent[x] == x else find(parent[x])
        for e in edges:
            x,y = map(find,e)
            if x == y: return False
            #2 nodes in e already share parent
            #if they are connected, there is a circle
            parent[x] = y
        return len(edges) == n-1 #an extra check for valid input

#------union find-----------------------------
#323. Number of Connected Components in an Undirected Graph
"""
 Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.

Example 1:

     0          3
     |          |
     1 --- 2    4

Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.

Example 2:

     0           4
     |           |
     1 --- 2 --- 3

Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]], return 1.

Note:
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges. 
"""
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        union_find = UnionFind(n)
        for edge in edges:
            union_find.union(edge[0], edge[1])
        return len(filter(lambda x: x != 0, union_find.sizes))

class UnionFind(object):
    
    def __init__(self, n):
        self.parents = range(n)
        self.sizes = [1] * n
    
    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            return self.find(self.parents[x])
    
    def union(self, x, y):
        
        find_x = self.find(x)
        find_y = self.find(y)
        if find_x == find_y:
            return True
        
        if self.sizes[find_x] <= self.sizes[find_y]:
            self.parents[find_x] = find_y
            self.sizes[find_y] += self.sizes[find_x]
            self.sizes[find_x] = 0
        else:
            self.parents[find_y] = find_x
            self.sizes[find_x] += self.sizes[find_y]
            self.sizes[find_y] = 0
#------union find-----------------------------
#200. Number of Islands
"""Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
You may assume all four edges of the grid are all surrounded by water.
Example 1:

11110
11010
11000
00000

Answer: 1

Example 2:

11000
11000
00100
00011

Answer: 3
"""
class UnionFind(object):
    
    def __init__(self, grid):
        # i*self.col + j
        self.row = len(grid)
        self.col = len(grid[0])
        self.count = 0
        self.parents = range( self.row*self.col ) #covert 2D grid to 1-D array
        for i in xrange(self.row):
            for j in xrange(self.col):
                if grid[i][j] == '1': self.count += 1 # max possibility of island
        self.sizes = [1] *(self.row*self.col )


    def find(self, x):
        while x!= self.parents[x]:
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x 
    
    def union(self, x, y):
        
        find_x = self.find(x)
        find_y = self.find(y)
        if find_x == find_y:  return 
        if find_x == find_y:
            return True
        if self.sizes[find_x] <= self.sizes[find_y]:
            self.parents[find_x] = find_y
            self.sizes[find_y] += self.sizes[find_x]
            self.sizes[find_x] = 0
        else:
            self.parents[find_y] = find_x
            self.sizes[find_x] += self.sizes[find_y]
            self.sizes[find_y] = 0
        
        self.count -= 1 #important! 



class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # for python, if input variable is array, it is mutable, i.e the value get changed after the function
        # call
        self.row = len(grid)
        if self.row ==0 : return 0
        self.col = len(grid[0])
        if self.col == 0: return 0
        return self.sol2(grid)
        
    def sol2(self,grid):
        #union find
        m = len(grid)
        n = len(grid[0])
        
        union_find = UnionFind(grid)
        for i in xrange(m):
            for j in xrange(n):
                if grid[i][j] == '0': continue
                p = i*n + j
                q = 0
                #if i> 0 and grid[i-1][j] == '1':
                #    q = p -n
                #    union_find.union(p,q)
                if i < m-1 and grid[i+1][j] == '1':
                    q = p+n
                    union_find.union(p,q)
                #if j >0 and grid[i][j-1] == '1':
                #    q = p-1
                #    union_find.union(p,q)
                if j < n-1 and grid[i][j+1] == '1':
                    q = p+1
                    union_find.union(p,q)
        return union_find.count 