from pdb import set_trace as bkp
#dfs method
def cyclic(g):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    complexity o
    """
    path = set()
    visited = set()
    # visited here is only used to skip the point, it use when we loop start point
    # path is used to check cycle, it's like the traditional visitied for one start
    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)

#------------
"""
L-- Empty list that will contain the sorted elements
S -- Set of all nodes with no incoming edges
while S is non-empty do
    remove a node n from S
    add n to tail of L
    for each node m with an edge e from n to m do
        remove edge e from the graph
        if m has no other incoming edges then
            insert m into S
if graph has edges then
    return error (graph has at least one cycle)
else 
    return L (a topologically sorted order)
"""
# study leetcode 210
#----------------------------
def bfs(g):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    complexity o
    """
    #similar like this, the code below cannot be used directly
    #but need to loop the input to find the numOfItems first
    degrees = [0 for i in xrange(numCourses)]
    childs = [[] for i in xrange(numCourses)]
    for pair in prerequisites:
        degrees[pair[0]]+=1
        childs[pair[1]].append(pair[0])
    A = set(range(numCourses)) #courses
    
    delqueue = []
    ans = []
    
    # find all courses that do not need prerequist
    for course in A:
        if degrees[course] == 0:
            delqueue.append(course)
            
    while delqueue:
        course = delqueue.pop(0)

        A.remove(course)
        
        for child in childs[course]:
            degrees[child]-=1
            if degrees[child] == 0:
                delqueue.append(child)
    return len(A) != 0

g1 = bfs({1: (2,), 2: (3,), 3: (1,)})
print g1
g2 = bfs({1: (2,), 2: (3,), 3: (4,)})
print g2

#-------------------------------------
# detect circle in undirected graph
# check leetcode 261
#261. Graph Valid Tree
"""
 Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.

Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.

Hint:

    Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], what should your return? Is this case a valid tree?
    According to the definition of tree on Wikipedia: “a tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.”

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
            parent[x] = y
        return len(edges) == n-1
        
    def sol2(self,n,edges):
        #dfs , stack method
        if len(edges) != n-1: return False
        neighbors = {i:[] for i in xrange(n)}
        for v,w in edges:
            neighbors[v].append(w)
            neighbors[w].append(v)
        #def visit(v):
        #    #pop v's value into a list and apply visit to it
        #    map(visit,neighbors.pop(v,[])) 
        #visit(0)
        stack = [0]
        while stack and neighbors:
            stack += neighbors.pop(stack.pop(), [])        
        return not neighbors
        
        """
        for iterative version, replace the 3 visit lines with 
        stack = [0]
        while stack:
            stack += neighbors.pop(stack.pop(), [])
        """
    
    def sol3(self,n,edges):
        #bfs method
        if len(edges) != n-1: return False
        neighbors = {i:[] for i in xrange(n)}
        for v,w in edges:
            neighbors[v].append(w)
            neighbors[w].append(v)

        queue = collections.deque([0])
        while queue and neighbors:
            queue.extend(neighbors.pop(queue.popleft(), []))     
        return not neighbors
        
    def sol4(self,n,edges):
        #topology sort
        #https://discuss.leetcode.com/topic/21869/a-python-solution-with-topological-sort
        #This solution looks like topological-sort, which iteratively removes the nodes with degree of 1.
        #The base condition is that a single node with no edges is a tree. By induction, if the graph is a tree, with the leaves removed, the rest part of it is still a tree
        graph = {i:set() for i in xrange(n)}
        for p, q in edges:
            graph[p].add(q)
            graph[q].add(p)
        while len(graph) > 0:
            leaves = list()
            for node, neighbors in graph.iteritems():
                if len(neighbors) <= 1:
                    leaves.append(node)
            if len(leaves) == 0:
                return False # a cycle exists
            for n in leaves:
                if len(graph[n]) == 0:
                    # must be one connected component
                    return len(graph) == 1 
                nei = graph[n].pop()
                graph[nei].remove(n)
                del graph[n]
        return True
         # Comment: A great example to topologically sort a undirected graph. 
         #For directed graph, we always start with nodes with 0 in-degree. 
         #For undirected graph, we first turn every undirected edge into two directed edges, and then start with nodes with 1 in-degree, or out-degree.