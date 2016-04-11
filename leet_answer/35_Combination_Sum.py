"""
Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
The solution set must not contain duplicate combinations.
For example, given candidate set 2,3,6,7 and target 7,
A solution set is:
[7]
[2, 2, 3]
"""
class Solution(object):
    def combinationSum(self, C, T):
        """
        :type C: List[int]
        :type T: int
        :rtype: List[List[int]]
        """
        # C, candidates need to be sort
        C.sort()
        self.ret = []
        self.lenc = len(C)
        self.dfs(0,[],C,T)
        return self.ret

    def dfs(self,start,vlist,C,T):
        if T == 0:
            self.ret.append(vlist)
            return
        for i in xrange(start,self.lenc):
            if C[i] >  T: continue
            if i< self.lenc-1 and C[i] == C[i+1]: continue
            self.dfs(i,vlist+[C[i]],C,T-C[i])