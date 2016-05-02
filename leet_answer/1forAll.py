#-------hash table-----------------------------
"""
Given an array of integers, return indices of the two numbers such that 
they add up to a specific target.
You may assume that each input would have exactly one solution.
:type nums: List[int]
:type target: int
:rtype: List[int]
"""
class Solution1(object):
    def twoSum(self, nums, target):
        # add one cmd
        tb = {}
        for i in xrange(len(nums)):
            tmp = target - nums[i]
            if tmp in tb:
                return [tb[tmp],i]
            else:
                tb[nums[i]] = i
        
#------list-----------------------------
"""
You are given two linked lists representing two non-negative numbers.
The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
"""
class Solution2(object):

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        return self.sol2(l1,l2)
    def sol2(self,l1,l2):
        dumy = ListNode(None)
        curr = dumy
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            val = (v1 + v2 +carry)%10
            carry = (v1 + v2 +carry)/10
            curr.next = ListNode(val)
            curr = curr.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return dumy.next

#-----hash and 2 pointer------------------------------
"""
Given a string, find the length of the longest substring without repeating characters.

Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3.
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
:type s: str
:rtype: int
"""
class Solution3(object):

    def lengthOfLongestSubstring(self, s):

        if len(s) <=1 : return len(s)
        tb = {}
        start = -1
        end = 0
        ret = 0
        for i in xrange(len(s)):
            if s[i] not in tb:
                tb[s[i]] = i
                end = i
            else:
                currlen = end - start 
                ret = max(ret,currlen)
                tmp = tb[s[i]]
                for j in xrange(start+1,tmp+1):
                    del tb[s[j]]
                start = tmp
                end = i
                tb[s[i]] = i
        
        currlen = end - start 
        ret = max(ret,currlen)
        return ret
#---------binary search, divide and con--------------------------
# Median of Two Sorted Arrays
"""
There are two sorted arrays nums1 and nums2 of size m and n respectively. 
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
"""
class Solution4(object):
    def helper(self,A,B,K):
        # index 0 system, kth item is array[k-1]
        lenA = len(A); lenB = len(B)
        if lenA > lenB : return self.helper(B,A,K)
        if len(A) == 0: return B[K-1]
        if K  ==1 : return min(A[0],B[0])
        pa= min(K/2,lenA)
        pb = K-pa
        if A[pa-1] < B[pb-1]:
            return self.helper(A[pa:],B,pb)
        else:
            return self.helper(A,B[pb:],pa)

    def findMedianSortedArrays(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: float
        """
        lenA = len(A); lenB = len(B)
        if lenA == 0 and lenB == 0: return False
        if (lenA + lenB)%2 ==1:
            return self.helper(A,B,(lenA+lenB)/2 +1)
        else:
            return (self.helper(A,B,(lenA+lenB)/2 +1) + self.helper(A,B,(lenA+lenB)/2))*0.5
#-----------------------------------
"""
Given a string S, find the longest palindromic substring in S.
You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic subs
"""
#5. Longest Palindromic Substring
class Solution5(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        return self.sol3(s)
        
    def sol3(self,s):
        #Manacher's Algorithm
        T = self.process(s)
        C=R=0 #center and right boundary
        P=[0]*len(T) # counter for each position
        for i in xrange(1,len(T)-1):
            i_mirror = 2*C-i
            if R>i:
                P[i] = min(R-i,P[i_mirror])
            else:
                P[i] = 0
                
            while T[i+1+P[i]] ==  T[i-1-P[i]]: P[i] = P[i]+1
            if i+P[i] > R:
                C = i
                R = i+P[i]
                
        maxLen = 0
        centerIndex = 0
        for i in range(1, len(T)-1):
            if P[i] > maxLen:
                maxLen = P[i]
                centerIndex = i
        return s[(centerIndex-1-maxLen)/2: (centerIndex-1-maxLen)/2+maxLen]
        
        
    def process(self,s):
        """
        ^ head $tail #seperator
        """
        ret = "^"
        for i in xrange(len(s)):
            ret+="#"+s[i]
        ret += "#$"
        return ret
        

    def dp(self,s):
        # time o(n^2), space o(n^2)
        # d[i][j] == True means s[i] to s[j] is a palindrome 
        n = len(s)
        longbegin = 0
        maxlen =1
        d = [[False for j in xrange(n)] for i in xrange(n)]
        for i in xrange(n): d[i][i] = True
        for i in xrange(n-1):
            if s[i] == s[i+1]: 
                d[i][i+1] = True
                longbegin = i
                maxlen = 2
        for l in xrange(3,n+1):  #l the length of palindrome string
            for i in xrange(n-l):
                j = i+l
                if s[i]==s[j] and d[i+1][j-1]:
                    d[i][j]=True
                    longbegin = i
                    maxlen = l
        return s[longbegin:longbegin+maxlen]
        
    def sol1(self,s):
        # solution1 O(n**2) and o(1) space
        # imporvement of brute force
        ret = ''
        for i in xrange(len(s)):
            # s1 and s2 because the palindrome string center can be i or bewtween i and i+1
            s1 = self.getlps(s,i,i)
            if len(s1) > len(ret) : ret = s1
            s2 = self.getlps(s,i,i+1)
            if len(s2) > len(ret) : ret = s2
        return ret
        
        
    def getlps(self,s,l,r):
        while l >=0  and r < len(s) and s[l] == s[r]:
            l -=1
            r +=1
        return s[l+1:r]
#-----------------------------------
"""
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this
"""
#6. ZigZag Conversion
class Solution6(object):
    def convert(self, s, n):
        if n < 1: return None
        if n ==1 or n >= len(s): return s
        return self.sol2(s,n)

        
    def sol2(self,s,n):
        arr = ["" for i in xrange(n)]
        row =0;step =1
        for c in s:
            arr[row] += c
            if row == 0: step =1
            elif row == n-1: step = -1
            row += step
        return "".join(arr)
        
        
    def sol1(self,s,n):
        # keypoint is r = 2*n-2
        # arr store string in each level
        arr = ["" for i in xrange(n)]
        r = 2*n-2
        for i in xrange(len(s)):
            tmp = i%r
            if tmp < n:
                arr[tmp] += s[i]
            else:
                arr[r-tmp] += s[i]
        return "".join(arr)
#-----------------------------------
#7. Reverse Integer
"""
Reverse digits of an integer.
"""
class Solution7(object):
    def reverse(self, x):
        #  INT_MAX = 2147483647
        #  INT_MIN = -2147483648
        intmax = 2**31-1
        intmin = -2**31
        maxd10 = intmax/10
        ret = 0
        sign = 1 if x>0 else -1
        x = abs(x)
        while x > 0:
            digit = x%10
            if ret > maxd10 or ret == maxd10 and digit >7: return 0
            ret = ret*10 + digit
            x /=10
        ret = sign * ret
        return ret
#-----------------------------------
#8. String to Integer (atoi)
"""
Implement atoi to convert a string to an integer.

Hint: Carefully consider all possible input cases.
If you want a challenge, please do not see below and ask yourself what are the possible input cases.

Notes: It is intended for this problem to be specified vaguely (ie, no given input specs).
You are responsible to gather all the input requirements up front. 
"""
class Solution8(object):
    # note this solition did not mention that we can have 'e' in the string
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        #INT_MAX = 2147483647
        #INT_MIN = -2147483648
        str = str.strip()
        N = len(str)
        if N == 0:  return 0
        sign = 1
        res = 0
        imin, imax = -1<<31, (1<<31)-1
        maxd10 = imax/10
        for i, bit in enumerate(str):
            if i == 0 and bit in ['-', '+']:
                if bit == '-':
                    sign = -1
            elif bit.isdigit():
                if res > maxd10 or res == maxd10 and int(bit) >7:
                    return imax if sign ==1 else imin
                res = res*10 + int(bit)
            else:
                break
        return sign * res
#-----------------------------------
#9.Palindrome Number
class Solution9(object):
    def isPalindrome(self, x):
        if (x < 0) or (x!=0 and x%10 == 0): return False
        return self.sol2(x)
        
    def sol2(self,x):
        div = 1
        while (x/div >=10):
            div *= 10
        while x!=0:
            l = x/div
            r = x%10
            if l!=r: return False
            x = (x%div)/10
            div /= 100
        return True
    def sol1(self,x):
        # half reverse
        # stop at the half since reverse the whole number could cause overflow
        sum = 0
        while x > sum:
            sum =10 *sum + x%10
            x /= 10
        return sum == x or sum/10 == x
#-----db------------------------------
#10. Regular Expression Matching
"""
Implement regular expression matching with support for '.' and '*'.
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
"""
class Solution10(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        return self.sol1(s,p)
        #return self.dfs(s,p)
    
    def sol1(self,s,p):
        #dp method
        # Let d(i, j) = true if s[0, i - 1] matches p[0, j - 1] (i, j are string lengths).
        #Initialize:
        #d(0, 0) = true, 
        #d(0, j): if p[j - 1] == '*', d(0,j) = d(0, j - 2) // deletion; else d(j) = false.
        #Fill up the table:
        #if         p[j - 1] matches s[i - 1],   d(i, j) = d(i - 1, j - 1);
        #else if  p[j - 1] == '*',  two cases:
        #       if  p[j - 2] matches s[i - 1],   d(i, j) = deletion: d(i, j - 2) || repetition: d(i - 1, j);
        #       else                                        d(i, j) = deletion: d(i, j - 2);
        #Note: “p[j] matches s[i]” means p[j] == s[i] || p[j] == '.'.
        ls =len(s)
        lp = len(p)
        dp = [[False for j in range(len(p) + 1)] for i in range(len(s) + 1)]
        dp[0][0] = True
        #!! very important to set dp[0][j]
        # this is for the case like p == "a*"
        for j in range(1,len(p) + 1):
                if p[j-1] == '*' and j >= 2:  
                        dp[0][j] = dp[0][j - 2]
        for i in xrange(1,ls+1):
            for j in xrange(1,lp+1):
                if p[j - 1] == '.' or s[i-1] == p[j-1]:   ## first case
                    dp[i][j] = dp[i - 1][j - 1] 
                elif p[j-1] == "*" :
                    if p[j-2] == '.' or s[i-1] == p[j-2]:
                        dp[i][j] = dp[i][j-2] or dp[i-1][j]  #delete x*to match or the * is used for repete
                    else:
                        dp[i][j] = dp[i][j-2]  #delete the x* to match
        return dp[-1][-1]

#---------two pointer--------------------------
#11. Container With Most Water
"""
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container. 
"""
class Solution11(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        lh = len(height)
        if lh <= 1 : return 0
        ret =0
        left = 0
        right = lh -1
        while left < right:
            if height[left] < height[right]:
                area = height[left] * (right - left)
                left +=1
            else:
                area = height[right] * (right - left)
                right -=1
            if area > ret:
                ret = area
        return ret
#-----------------------------------
#12. Integer to Roman 
"""
Given an integer, convert it to a roman numeral.

Input is guaranteed to be within the range from 1 to 3999.
"""
class Solution12(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        digits = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD' ),
                  (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                  (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
        result = ""
        for digit in digits:
            while num >= digit[0]:
                result += digit[1]
                num -= digit[0]
            if num == 0:
                break
        return result
#-----------------------------------
#13. Roman to Integer
"""
Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.
"""
class Solution13(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) ==0 : return None
        tb = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        p,q = 0,1
        res = 0
        while q < len(s):
            if tb[s[p]] < tb[s[q]]:
                res -= tb[s[p]]
            else:
                res += tb[s[p]]
            p,q = q,q+1
        res += tb[s[p]]
        return res
#-----------------------------------
class Solution14(object):
#-----------------------------------
#15 3Sum
"""
Given an array S of n integers, are there elements a, b, c in S 
such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
"""
class Solution15(object):
    def threeSum(self, A):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(A) < 3: return []
        res = list()
        A.sort()
        length = len(A)
        for i in xrange(length-2):
            j = i +1
            k = length-1
            while (j < k):
                if A[j]+A[k] == -1*A[i]:
                    res.append([A[i],A[j],A[k]])
                    j+=1
                    k-=1
                elif A[j]+A[k] < -1*A[i]:
                    j = j+1
                else:
                    k = k-1
        return [list(x) for x in set([tuple(y)for y in res])]
                    
#-----------------------------------
class Solution16(object):
#-----------------------------------
class Solution17(object):
#-----------------------------------
#-----------------------------------
#-----------------------------------
#--------2 pointers---------------------------
#28. Implement strStr()
"""
 Implement strStr().

Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Subscribe to see which companies asked this question

"""
class Solution28(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        # 2 pointer solution
        if haystack == needle or needle == "" : return 0
        if not haystack: return -1
        lh = len(haystack)
        ln = len(needle)
        if ln > lh : return -1
        for i in xrange(lh-ln+1):
            for j in xrange(ln):
                if haystack[i+j] != needle[j]:
                    break
                if j == ln-1: return i
                
        return -1
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#39 Combination Sum
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
class Solution39(object):
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
#-----------------------------------
#40 Combination Sum II
"""
Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
The solution set must not contain duplicate combinations.
For example, given candidate set 10,1,2,7,6,1,5 and target 8,
A solution set is:
[1, 7]
[1, 2, 5]
[2, 6]
[1, 1, 6]
"""
class Solution40(object):
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum2(self, C, T):
        """
        :type C: List[int]
        :type T: int
        :rtype: List[List[int]]
        """
        C.sort()
        self.ret = []
        self.lenc = len(C)
        self.dfs(0,[],C,T)
        return self.ret
        
    def dfs(self,start,vlist,C,T):
        if T == 0 and vlist not in self.ret:
            self.ret.append(vlist)
            return
        for i in xrange(start,self.lenc):
            if C[i] >  T: continue
            self.dfs(i+1,vlist+[C[i]],C,T-C[i])

#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#--------dfs---------------------------
#98. Validate Binary Search Tree
"""
 Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.

"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.dfs(root,-1*float('inf'),float('inf'))
    def dfs(self,root,min,max):
        if root == None: return True
        if root.val <= min or root.val >= max: return False
        return   self.dfs(root.left,min,root.val) and  self.dfs(root.right,root.val,max)
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#215 Kth Largest Element in an Array
"""
Find the kth largest element in an unsorted array. Note that it is the kth largest 
element in the sorted order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5. 
"""
import random
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # method 2 divide and conquer
        pivot = random.choice(nums)
        nums1,nums2 = [],[] #big ,small
        for num in nums:
            if num > pivot:
                nums1.append(num)
            elif num < pivot:
                nums2.append(num)
        if len(nums1) >= k:
            return self.findKthLargest(nums1,k)
        elif k > len(nums)-len(nums2):
            return self.findKthLargest(nums2,k-(len(nums)-len(nums2)))
        return pivot

    def sol1(self, nums, k):
        # method 1 using heap
        #http://wlcoding.blogspot.com/2015/05/kth-largest-element-in-array.html?view=sidebar
        if len(nums) < k : return False
        if len(nums)== 0 : return False
        h = []
        ret = nums[0]
        for i in xrange(len(nums)):
            if i < k:
                heapq.heappush(h,nums[i])
            else:
                ret = heapq.heappop(h)
                if nums[i] > ret: heapq.heappush(h,nums[i])
                else: heapq.heappush(h,ret)

        return heapq.heappop(h)