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
    # visited here is only used to skip the point
    # path is used to check cycle
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