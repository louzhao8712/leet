#-------hash table-----------------------------
#1 Two Sum
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
#14. Longest Common Prefix
"""
Write a function to find the longest common prefix string amongst an array of strings. 
"""
class Solution14(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0: return ''
        #base str is strs[0]
        retlen = len(strs[0])
        for st in strs[1:]:
            i = 0
            while i < retlen and i < len(st) and st[i]==strs[0][i]:
                i +=1
            retlen = min(retlen,i)
        return strs[0][0:retlen]

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
#-----------------------------------
#19. Remove Nth Node From End of List
"""
Given a linked list, remove the nth node from the end of list and return its head.
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(None)
        dummy.next = head
        prev = curr = dummy
        for i in xrange(n):
            curr = curr.next
            if curr.next == None and i!=n-1:
                return dummy.next
        while curr.next:
            prev = prev.next
            curr = curr.next
        prev.next=prev.next.next
        return dummy.next
#-----------------------------------
#-----------------------------------
#-----------------------------------
#26. Remove Duplicates from Sorted Array
"""
 Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

Subscribe to see which companies asked this question

"""
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ln = len(nums)
        if ln <=1 : return ln
        pos =0
        for i in xrange(1,ln):
            if nums[i] != nums[pos]:
                pos += 1
                nums[pos] = nums[i]
        return pos + 1
#-----------------------------------
#27. Remove Element
"""
Given an array and a value, remove all instances of that value in place and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.
"""
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if not nums: return 0
        ret = 0
        for i in range(0,len(nums)):
            if nums[i] == val:
                continue
            else:
                nums[ret]=nums[i]
                ret +=1
        # nums[:ret]
        return ret
#--------2 pointers---------------------------
#28. Implement strStr()
"""
 Implement strStr().

Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Subscribe to see which companies asked this question

"""
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        # Robin-Karp method
        """
        T test, P pattern
        """
        T = haystack
        P = needle
        if T == P or P == "" : return 0
        if not T: return -1
        lt = len(T)
        lp = len(P)
        if lt < lp: return -1

        prime = 1000000007#big prime
        x = 29 # random(1,prime-1)
        result = []
        pHash = self.PolyHash(P,prime,x)
        
        # precomputeHashes
        H = [None for i in range(lt-lp+1)]
        S = T[lt-lp:lt]
        H[lt-lp] = self.PolyHash(S,prime,x)
        y =1
        for i in range(1,lp+1):
            y = (y*x) %prime
        for i in range(lt-lp-1,-1,-1):
            H[i] = (x*H[i+1]+ord(T[i])-y*ord(T[i+lp]))%prime
       
        for i in range(lt-lp+1):
    
            if pHash != H[i]: continue
            if T[i:i+lp] == P: return i
        return -1

    def PolyHash(self,P,prime,x):
        ans = 0
        for c in reversed(P):
            ans = (ans * x + ord(c)) % prime
        return ans % prime
        
    def sol1(self, haystack, needle):
        """
        native method
        """
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
#45. Jump Game II
"""
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps. 
"""
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # jums : the minimum number of jumps
        # prev: the fathest index that we have reached
        # maxv : the fathest index that we potentially reach
        # when i > prev: we need to jump, but we don't have to jump from i, we can backward and jump from the 
        # index which can reach maxv
        # for each i update the maxv
        jumps = prev = maxv = 0
        for i in xrange(len(nums)):
            if i > prev:
                if maxv == prev: return -1
                prev = maxv
                jumps +=1
                if prev == len(nums)-1: break
            maxv = max(maxv,i+nums[i])
        return jumps
#-----------------------------------
#-----------------------------------
#-----------------------------------
#55. Jump Game
"""
 Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index. 
"""
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return self.sol2(nums)

    def sol2(self,nums):
        # jumps : the minimum number of jumps
        # prev: the fathest index that we have reached
        # maxv : the fathest index that we potentially reach
        # when i > prev: we need to jump, but we don't have to jump from i, we can backward and jump from the 
        # index which can reach maxv
        # for each i update the maxv
        jumps = prev = maxv = 0
        for i in xrange(len(nums)):
            if i > prev:
                if maxv == prev: return False
                prev = maxv
                jumps +=1
                if prev == len(nums)-1: break
            maxv = max(maxv,i+nums[i])
        return True

    def sol1(self,nums):
        d =0
        for i in xrange(1,len(nums)):
            d = max(d,nums[i-1])-1
            if d < 0: return False
        return d >= 0
#-----------------------------------
#56. Merge Intervals
"""
Given a collection of intervals, merge all overlapping intervals.

For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18]. 
"""

#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, inters):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        inters.sort(key= lambda x:x.start)
        n = len(inters)
        res = []
        for i in xrange(n):
            if res == []:
                res.append(inters[i])
            else:
                if res[-1].start <= inters[i].start <= res[-1].end:
                    res[-1].end = max(res[-1].end, inters[i].end)
                else:
                    res.append(inters[i])
        return res    
#62. Unique Paths
"""
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
"""
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        return self.sol3(m,n)
        
    def sol1(self,m,n):
        if m < 1 or n < 1 : return None
        import math
        a = max(m-1,n-1)
        b= min(m-1,n-1)
        tot = m+n-2
        #Ctot-a
        return math.factorial(tot)/(math.factorial(b) *math.factorial(a))
        
        
    def sol2(self,m,n):
        if m < 1 or n < 1 : return None
        st = [[0 for i in xrange(n)] for j in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if i== 0 or j ==0 : st[i][j] = 1
                else:
                    st[i][j] = st[i-1][j]+st[i][j-1]
        return st[-1][-1]
        
    def sol3(self,m,n):
        #dp but save memory
        minv = min(m,n)
        maxv = max(m,n)
        st = [0 for i in xrange(minv)]
        for i in xrange(maxv):
            for j in xrange(minv):
                if i== 0 or j ==0 : st[j] = 1
                else:
                    st[j] = st[j]+st[j-1]
        return st[-1] 
#-----------------------------------
#63. Unique Paths II
"""
Follow up for "Unique Paths":

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

For example,
"""
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        
        if obstacleGrid[0][0] == 1: return 0
        m= len(obstacleGrid)
        n = len(obstacleGrid[0])
        if m < 1 or n < 1 : return None
        list = [[0 for i in range(n)] for i in range(m)]

        for i in range(0, m):
            for j in range(0, n):
                if obstacleGrid[i][j]==1: 
                    list[i][j] = 0
                    continue
                if i==0 and j==0: list[i][j] =1
                else:
                    list[i][j] = \
                        (0 if i ==0 else list[i-1][j]) + \
                        (0 if j ==0 else list[i][j-1]) 
        return list[m-1][n-1]
#-----------------------------------
#64.Minimum Path Sum
"""
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
"""
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return self.sol2(grid)
        
    def sol1(self,grid):
        h = len(grid)
        w = len(grid[0])
        st = [[0 for i in xrange(w)] for j in xrange(h)]
        for i in xrange(h):
            for j in xrange(w):
                if i== 0 and j ==0 : st[i][j] = grid[i][j]
                elif i == 0: st[i][j] = st[i][j-1] + grid[i][j]
                elif j == 0: st[i][j] = st[i-1][j] + grid[i][j]
                else:
                    st[i][j] = min(st[i-1][j],st[i][j-1]) + grid[i][j]
        return st[-1][-1] 
        
    def sol2(self,grid):
        #1D dp
        h = len(grid)
        w = len(grid[0])
        if h==0 or w==0: return 0
        st = [float('inf') for i in xrange(w)]
        st[0] = 0
        for i in xrange(h):
            st[0] = st[0] + grid[i][0]
            
            for j in xrange(1,w):
                st[j]= min(st[j-1],st[j])+grid[i][j]

        return st[w-1]
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#70. Climbing Stairs
"""
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top? 
"""
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        return self.sol2(n)
    def sol2(self,n):
        if n<=1: return n
        a =b =c=1
        for i in xrange(2,n+1):
            c= a +b
            a = b
            b =c
            
        return c        
        
    def sol1(self,n):
        if n<=1: return n
        dp = [0 for i in xrange(n+1)]
        dp[0]=1
        dp[1] =1
        
        for i in xrange(2,n+1):
            dp[i] = dp[i-1] +dp[i-2]
        return dp[n]
#----------------------------------
#-----------------------------------
#-------dp----------------------------
#73 Edit Distance
"""
 Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character
"""
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # solution1 2d dp
        #Let d(i, j) be distance between str1 (len = i) and str2 (len = j) 
        #if min{i, j} = 0, d(i, j) = max{i, j} 
        #else d(i, j) = min{ d(i - 1, j) + 1,  // add last char in str1
        #                          d(i, j - 1) + 1,  // add last char in str2
        #                          d(i - 1, j - 1) + (str1[i] == str2[j]) ? 0 : 1 // replace the last char in str1 with that in str2 when str1[i] != str2[j] }   
        n1 = len(word1)
        n2 = len(word2)
        
        dp = [ [ 0 for i in xrange(n2+1)] for j in xrange(n1+1)]
        for i in xrange(n1+1):
            for j in xrange(n2+1):
                if i==0 or j == 0: dp[i][j] = max(i,j)
                else:
                    dp[i][j] = min( dp[i-1][j]+1,\
                                    dp[i][j-1]+1,\
                                    dp[i-1][j-1]+ (0 if word1[i-1]==word2[j-1] else 1))
        return dp[n1][n2]
#----binary search------------------
#74. Search a 2D Matrix
"""
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.

For example,
"""
class Solution(object):
    def searchMatrix(self, matrix, target):
        return self.sol2(matrix,target)

    def sol1(self, matrix, target):
        # Find the first position of target
        if not matrix or not matrix[0]:
            return False
        m, n = len(matrix), len(matrix[0])
        st, ed = 0, m * n - 1

        while st + 1 < ed:
            mid = (st + ed) / 2
            if matrix[mid / n][mid % n] == target:
                return True
            elif matrix[mid / n][mid % n] < target:
                st = mid
            else:
                ed = mid
        return matrix[st / n][st % n] == target or \
                matrix[ed / n][ed % n] == target
            
    def sol2(self, matrix, target):
        if not matrix or not matrix[0]:
            return False

        # first pos >= target
        st, ed = 0, len(matrix) - 1
        while st + 1 < ed:
            mid = (st + ed) / 2
            if matrix[mid][-1] == target:
                st = mid
            elif matrix[mid][-1] < target:
                st = mid
            else:
                ed = mid
        if matrix[st][-1] >= target:
            row = matrix[st]
        elif matrix[ed][-1] >= target:
            row = matrix[ed]
        else:
            return False

        # binary search in row
        st, ed = 0, len(row) - 1
        while st + 1 < ed:
            mid = (st + ed) / 2
            if row[mid] == target:
                return True
            elif row[mid] < target:
                st = mid
            else:
                ed = mid
        return row[st] == target or row[ed] == target
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#83 Remove Duplicates from Sorted List 
"""
 Given a sorted linked list, delete all duplicates such that each element appear only once.

For example,
Given 1->1->2, return 1->2.
Given 1->1->2->3->3, return 1->2->3. 
"""
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(None)
        dummy.next = head
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val: curr.next = curr.next.next
            else: curr = curr.next

        return dummy.next
#-----------------------------------
#-----------------------------------
#-----------------------------------
#91. Decode Ways
"""
A message containing letters from A-Z is being encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits, determine the total number of ways to decode it. 
"""
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        return self.sol2(s)
        
    def sol1(self,s):
        n = len(s)
        if n == 0: return 0
        if s[0] == '0' : return 0
        prev =1
        curr =1
        for i in xrange(2,n+1):
            tmp = 0
            if s[i-1]!= '0' : tmp += curr
            if s[i-2] == '1' or (s[i-2] == '2' and int(s[i-1]) <= 6):
                tmp += prev
                
            prev = curr
            curr = tmp
        return curr
        
        
    def sol2(self,s):
        #dp[i]表示前i-1个数字的DW dp[i] is the ways to decode s[0:i]
        if s=="" or s[0]=='0': return 0
        dp=[1,1]
        for i in range(2,len(s)+1):
            if 10 <=int(s[i-2:i]) <=26 and s[i-1]!='0':
                dp.append(dp[i-1]+dp[i-2])
            elif int(s[i-2:i])==10 or int(s[i-2:i])==20:
                dp.append(dp[i-2])
            elif s[i-1]!='0':
                dp.append(dp[i-1])
            else:
                return 0
        return dp[len(s)]
#-----------------------------------
#-----------------------------------
#-----------------------------------
#94Binary Tree Inorder Traversal
"""
Given a binary tree, return the inorder traversal of its nodes' values.
"""
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # recursive method
        self.ans = []
        self.iterative(root)
        return self.ans
    def iterative(self,root):
        stack = []
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                top = stack.pop()
                self.ans.append(top.val)
                root = top.right
    def recursive(self,root):
        if root:
            self.recursive(root.left)
            self.ans.append(root.val)
            self.recursive(root.right)
#-----------------------------------
#-----------------------------------
#---------dp--------------------------
#97. Interleaving String
"""
 Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

For example,
Given:
s1 = "aabcc",
s2 = "dbbca",

When s3 = "aadbbcbcac", return true.
When s3 = "aadbbbaccc", return false.

state: dp[i][j]表示s1的前i个字符配上s2的前j个字符在s3的前i+j个字符是不是IS
function:

dp[i][j] = True  # if dp[i-1][j] and s1[i-1] == s3[i-1+j]
         = True  # if dp[i][j-1] and s2[j-1] == s3[i+j-1]
         = False # else
initialize: dp[0][0] = True
answer: dp[M][N]
"""
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        #dp[i][j] means s1[0:i](len==i, last element is i-1) and s2[0:j] (len j) can create interleve string s3[0:i+j] (len i+j)
        ls1 = len(s1)
        ls2 = len(s2)
        ls3 = len(s3)
        if ls3 != ls1+ls2: return False
        dp = [[False for j in xrange(ls2+1)] for i in xrange(ls1+1)]
        for i in xrange(ls1+1):
            for j in xrange(ls2+1):
                if i==0 and j==0: 
                    dp[0][0] = True
                elif i==0:
                    dp[0][j] = dp[0][j-1] and s2[j-1]==s3[j-1]
                elif j==0:
                    dp[i][0] = dp[i-1][0] and s1[i-1]==s3[i-1]
                else:
                    dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or\
                                (dp[i][j-1] and s2[j-1] == s3[i+j-1])
        return dp[ls1][ls2]
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

class Solution98(object):
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
#101. Symmetric Tree
"""
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        #if root == None: return True
        #return self.dfs(root.left,root.right)
        return self.iterative(root)
        
    def iterative(self,root):
        """
        Instead of recursion, we can also use iteration with the aid of a queue. Each two consecutive nodes in the queue should be equal, and their subtrees a mirror of each other. Initially, the queue contains root and root. Then the algorithm works similarly to BFS, with some key differences. Each time, two nodes are extracted and their values compared. Then, the right and left children of the two nodes are inserted in the queue in opposite order. The algorithm is done when either the queue is empty, or we detect that the tree is not symmetric (i.e. we pull out two consecutive nodes from the queue that are unequal).
        """
        if root == None: return True
        q = []
        q.append(root.left)
        q.append(root.right)
        while q:
            t1 = q.pop(0)
            t2 = q.pop(0)
            if t1 == None and t2 == None: continue
            if t1 == None or t2 == None: return False
            if t1.val!=t2.val: return False
            q.append(t1.left)
            q.append(t2.right)
            q.append(t1.right)
            q.append(t2.left)
        return True
        
        
    def dfs(self,p,q):
        if p== None and q == None : return True
        if p == None or q == None or p.val!=q.val : return False
        return self.dfs(p.left,q.right) and self.dfs(p.right,q.left)
#-----------------------------------
#-----------bfs------------------------
#103. Binary Tree Zigzag Level Order Traversal
"""
Given a binary tree, return the zigzag level order traversal of its nodes' values.
(ie, from left to right, then right to left for the next level and alternate between).
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]

        """
        self.res = []
        #self.dfs(root,0)
        self.bfs(root)
        return self.res

    def bfs(self,root):
        if root == None: return
        queue = [root,'end']
        while queue:
            tmplist = []
            curr = queue.pop(0)
            while curr!= 'end':
                tmplist.append(curr.val)
                if curr.left: queue.append(curr.left)
                if curr.right: queue.append(curr.right)
                curr = queue.pop(0)
            if len(self.res)%2 == 1: tmplist.reverse()
            self.res.append(tmplist)
            if queue: queue.append('end')

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
#114 Flatten Binary Tree to Linked List
"""
Given a binary tree, flatten it to a linked list in-place. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # connect the rightmost node in curr.left to curr.right
        # set curr.right to curr.right
        # set curr.left to None
        curr = root
        while curr:
            if curr.left:
                if curr.right: #connect the rightmost node in curr.left to curr.right
                    next = curr.left
                    while next.right:
                        next = next.right
                    next.right = curr.right
                
                curr.right = curr.left
                curr.left = None
                    
            curr = curr.right
"""
#### Flatten BST to (Doubly) linked list
1. Leetcode上面的原题是to single, 但是traversal是pre-order
2. 这里的doubly用的方法是in-order traversal, pre-order也是一样的思路
3. [网上](http://cslibrary.stanford.edu/109/TreeListRecursion.html)的题目还有点差别是要变成Circular Doubly Linked List
4. 稍微注意一下return的问题, 这两种recursion的方法都没有return值, 所以如果需要找head的话还得再处理下
5. 千万记得这里需要用到global declaration

#####Flatten思路
1. 最方便的方法还是用recursion
2. 先弄清需要的是preorder, inorder还是postorder的顺序
3. 选择对应order的traversal模板, 重要的一点是要把
   ```python
   left = root.left
   right = root.right
   ```
   提前存好，因为进行flatten之后可能会破坏树的结构，这步做好之后，XXXorder traversal的方法都是一样的了
4. 记得```global head, last```然后对```last```进行操作
   * Singly Linked List - 记得重置```last.left = None, last.right = root```
   * Doubly Linked List - 如果```last.right = root, root.left = last```
     这里有一点点差别就是如果是preorder的话，```head.left = None```需要单独处理下
5. ```last = root```更新```last```
6. ```head```就是初始设为None, 第一个需要处理的node就赋为```head```就行了

"""
#last = None
#head = None
def inorder_doubly_flatten(root):
    global last
    global head
    if not root:
        return
    inorder_doubly_flatten(root.left)
    if last:
        last.right = root
        root.left = last
    last = root
    if not head:                        # Used to get true HEAD
        head = root
    inorder_doubly_flatten(root.right)


#last = None
#head = None
def preorder_doubly_flatten(root):
    if not root:
        return
    global last
    global head
    right = root.right
    left = root.left
    if not head:
        head = last
    if last:
        last.right = root
        root.left = last
    else:
        root.left = None                # 小处理
    last = root

    preorder_doubly_flatten(left)
    preorder_doubly_flatten(right)
#-------dp----------------------------
#115. Distinct Subsequences
"""
 Given a string S and a string T, count the number of distinct subsequences of T in S.

A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).

Here is an example:
S = "rabbbit", T = "rabbit"

Return 3.
"""
class Solution(object):
    def numDistinct(self, S, T):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        return self.sol2(S,T)

    def sol1(self,S,T):
        #http://www.cnblogs.com/asrman/p/4009924.html
        #dp[i][j] means s[0:i] has how many subdistinct which is t[0:j]
        #http://www.cnblogs.com/asrman/p/4009924.html
        #dp[0][j] == 0
        """
        大概意思就是， 因为算的是S的子串和T匹配的方法， 所以一旦S[:j-1]和T[:i]有x种匹配方法时
        S[:j]必定也至少和T[:i]有x种匹配方法，但尤其当S[j-1]==T[i-1]的时候，需要再加上S[:j-1]和T[:i-1]的匹配方法数
        注意分清M,i和N,j对应T和S，这个很特殊因为必须是S的子串和T相同
        """
        dp = [ [0 for j in range(len(T) + 1)] for i in range(len(S) + 1) ]
        for i in range(len(S) + 1):
            dp[i][0] = 1
            
        for i in range(1, len(S) + 1):
            for j in range(1, len(T) + 1):
                if S[i - 1] == T[j - 1]:
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[len(S)][len(T)]

    def sol2(self,S,T):
        #1D DP, similiar to sol1
        #dp[0][j] == 0
        dp = [0 for j in range(len(T) + 1)] 
        dp[0] =1
            
        for i in range(1, len(S) + 1):
            for j in range(len(T),0,-1): #very important, sweep from right to left, update dp[j] first so that it won't affect dp[j-1]
                if S[i - 1] == T[j - 1]:
                    dp[j] += dp[j-1]

        return dp[len(T)]
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#118. Pascal's Triangle
"""
Given numRows, generate the first numRows of Pascal's triangle.

For example, given numRows = 5,
"""
class Solution(object):
    def generate(self, n):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res =[]
        if n == 0: return []
        if n > 0 : res.append( [1])
        if n > 1 : res.append([1,1])
        for i in xrange(3,n+1):
            tmp = [1]
            for j in xrange(1,i-1):
                tmp.append(res[i-2][j-1]+res[i-2][j])
            tmp.append(1)
            res.append(tmp)
        return res
#-----------------------------------
#119. Pascal's Triangle II
"""
Given an index k, return the kth row of the Pascal's triangle.

For example, given k = 3,
Return [1,3,3,1]. 
"""
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        res =[]
        if rowIndex >= 0 : res = [1]
        if rowIndex >= 1 : res = [1,1]
        for i in xrange(2,rowIndex+1):
            tmp = [1]
            for j in xrange(1,i):
                tmp.append(res[j-1]+res[j])
            tmp.append(1)
            res = tmp
        return res
#-----------------------------------
#120. Triangle
"""
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
"""
class Solution(object):
    def minimumTotal(self, T):
        """
        :type T: List[List[int]]
        :rtype: int
        """
        # method2, add from bottom to up
        # maintain an sum array length equal to the bottom row in triangle
        N = len(T)
        sum = [0 for i in xrange(len(T[-1]))]
        for r in xrange(N-1,-1,-1):
            for c in xrange(r+1):
                if r == N-1:
                    sum[c] = T[r][c]
                else:
                    sum[c] = min(sum[c],sum[c+1]) + T[r][c]
        return sum[0]
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-------dfs----------------------------
#131. Palindrome Partitioning
"""
Given a string s, partition s such that every substring of the partition is a palindrome.
Return all possible palindrome partitioning of s. 
"""
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        self.ans = []
        self.dfs(s,[])
        return self.ans

    def dfs(self,s,vlist):
        if len(s) == 0 and vlist != []:
            self.ans.append(vlist)
            return
        for i in xrange(1,len(s)+1):
            if self.isPalindrome(s[:i]):
                self.dfs(s[i:],vlist+[s[:i]])

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        lens = len(s)
        for i in xrange(lens/2):
            if s[i] != s[lens-1-i]: return False
        return True
#--------dp---------------------------
#132. Palindrome Partitioning II
"""
 Given a string s, partition s such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of s.

For example, given s = "aab",
Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut. 
"""
class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        #http://www.cnblogs.com/zuoyuan/p/3758783.html
        # dp: how many palindrome substring from i to n-1, so the mincut is dp[0]-1
        n = len(s)
        dp = range(n,-1,-1) #max substrings have

        p = [[False for i in xrange(n)] for j in xrange(n)]
        for i in xrange(n-1,-1,-1):
            for j in xrange(i,n):
                if s[i]==s[j] and (j-i <3 or p[i+1][j-1]):
                    p[i][j] = True
                    dp[i] = min(dp[i],dp[j+1]+1)
        return dp[0]-1
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#139. Word Break
"""
 Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code". 
"""
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        return self.bfs(s,wordDict)
        
    def bfs(self, s, wordDict):
        # 将当前单词拆分为前后两半，若前缀可以在字典dict中找到，则将后缀加入队列
        q = [s]
        visitSet = set([s])
        while q:
            item = q.pop(0)
            # all items in this queue is suffix
            if item in wordDict: return True
            prefix = ''
            for c in item:
                prefix += c
                suffix = item[len(prefix):]
                if prefix in wordDict and suffix not in visitSet:
                    q.append(suffix)
                    visitSet.add(suffix)
        return False
        
    def dpsolution(self, s, wordDict):
        #dp[i] == True means str[0:i] is breakable
        #dp[i] = dp[j] == True and s[j:i] in wordDict
        #if len(wordDict) == 0 and s!="": return False
        n = len(s)
        dp = [False for i in xrange(n+1)]
        dp[0] = True
        for i in xrange(1,n+1):
            for j in xrange(0,i):
                if dp[j] and s[j:i] in wordDict: 
                    dp[i] = True
                    break
        return dp[n]
#-----------------------------------
#140
#-----------------------------------
#141. Linked List Cycle
"""
 Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space? 
"""
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        p0 = p1 = head
        while p1 and p1.next:
            p1 = p1.next.next
            p0 = p0.next
            if p0 == p1 : return True
        return False
#-----------------------------------
#144.Binary Tree Preorder Traversal
"""
Given a binary tree, return the preorder traversal of its nodes' values.
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # recursive method
        self.ans = []
        self.iterative(root)
        return self.ans
    def iterative(self,root):
        stack = []
        while root or stack:
            if root:
                self.ans.append(root.val)
                stack.append(root)
                root = root.left
            else:
                top = stack.pop()
                root = top.right
    def recursive(self,root):
        if root:
            self.ans.append(root.val)
            self.recursive(root.left)
            self.recursive(root.right)
#-----------------------------------
#145. Binary Tree Postorder Traversal 
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.ans = []
        self.iterative(root)
        return self.ans
    def iterative(self,root):
        stack = []
        while root or stack:
            if root:
                self.ans.append(root.val)
                stack.append(root)
                root = root.right
            else:
                top = stack.pop()
                root = top.left
        # reverse mirror preorder
        self.ans.reverse()

    def recursive(self,root):
        if root:
            
            self.recursive(root.left)
            self.recursive(root.right)
            self.ans.append(root.val)
#-----------------------------------
#146. LRU Cache
"""
 Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item. 
"""
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
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#166. Fraction to Recurring Decimal
"""
Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

For example,

    Given numerator = 1, denominator = 2, return "0.5".
    Given numerator = 2, denominator = 1, return "2".
    Given numerator = 2, denominator = 3, return "0.(6)".

"""
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if denominator == 0: return None
        sign = "-" if numerator*denominator < 0 else ""
        numerator= abs(numerator)
        denominator = abs(denominator)
        rl = [] #return list
        db = {}
        cnt = 0
        repstr = None
        while True:
            rl.append(numerator/denominator)
            cnt += 1
            tmp = numerator%denominator
            if tmp == 0: break
            numerator = 10*tmp
            if db.get(numerator) : # repeat start
                repstr = "".join([str(x) for x in rl[db.get(numerator):cnt]])
                break
            db[numerator] = cnt
        
        res = str(rl[0])
        if len(rl) > 1: res += "."
        if repstr:
            res += "".join([str(x) for x in rl[1:db.get(numerator)]]) + "(" + repstr + ")"
        else:
            res += "".join([str(x) for x in rl[1:]])
        return sign + res
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
#155. Min Stack
"""
 Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

    push(x) -- Push element x onto stack.
    pop() -- Removes the element on top of the stack.
    top() -- Get the top element.
    getMin() -- Retrieve the minimum element in the stack.

"""
class MinStack(object):
    # two stacks, stack1 is the reg stack, stack2 is the minstack
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 =[]
        

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.stack1.append(x)
        if self.stack2 == [] or x <= self.stack2[-1]:
            self.stack2.append(x)
        

    def pop(self):
        """
        :rtype: nothing
        """

        if self.stack2 != [] and self.stack1 != [] and self.stack1[-1] == self.stack2[-1]:
            self.stack2.pop()
        
        if self.stack1 != []:
            self.stack1.pop()
        

    def top(self):
        """
        :rtype: int
        """
        if self.stack1!=[]: return self.stack1[-1]
        else:              return None        

    def getMin(self):
        """
        :rtype: int
        """
        if self.stack2!=[]:return self.stack2[-1]
        else: return None
#-----------------------------------
#-----------------------------------
#-----------------------------------
#160. Intersection of Two Linked Lists
"""
Write a program to find the node at which the intersection of two singly linked lists begins.
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB: return None
        lenA = self.getlen(headA)
        lenB = self.getlen(headB)
        
        if lenA > lenB:
            dist = lenA-lenB
            for i in xrange(dist):
                headA = headA.next
        elif lenA < lenB:
            dist = lenB- lenA
            for i in xrange(dist):
                headB= headB.next
        
        ret = None
        while headA and headB:
            if headA == headB:
                return headA
            else:
                headA = headA.next
                headB = headB.next
        return ret
        
    def getlen(self,head):
        ret = 0
        while head:
            ret +=1
            head = head.next
        return ret
#-----------------------------------
#-----------------------------------
#-----------------------------------
#169. Majority Element
"""
Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.
"""
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0 : return None
        ret = nums[0] ; count =1
        for i in xrange(1,n):
            if count == 0:
                ret = nums[i]
                count =1
            elif nums[i] == ret:
                count +=1
            else:
                count -=1
        return ret
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#172. Factorial Trailing Zeroes 
"""
Given an integer n, return the number of trailing zeroes in n!.
"""
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        ans =0
        # num of 5 + num of 25 + num of 125
        while n>0:
            n = n/5
            ans += n
        return ans
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#191. Number of 1 Bits
"""
Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
"""
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return self.sol1(n)
        
    def sol1(self,n):
        ans = 0
        while n:
            n&=n-1
            ans +=1
        return ans
        
    def sol2(self,n):
        ans = 0
        for i in xrange(32):
            ans += 1&n
            n >>=1
        return ans
#-----------------------------------
#-----------------------------------
#-----------------------------------
#198. House Robber
"""
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
"""
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.sol2(nums)
        
    def sol2(self,nums):
        n = len(nums)
        if n == 0 : return 0
        if n <=2 : return max(nums)
        odd = nums[0]
        even = max(nums[1],nums[0])
        for i in xrange(2,n):
            if i%2 == 0:
                odd = max(odd+nums[i],even)
            else:
                even = max(even+nums[i],odd)
        return max(odd,even)
        
    def sol1(self,nums):
        n = len(nums)
        if n == 0 : return 0
        if n <=2 : return max(nums)
        dp = [0 for i in xrange(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[1],nums[0])
        
        for i in xrange(2,n):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[n-1]
#-----------------------------------
#-----------------------------------
#-----------------------------------
#206 Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        #return self.recursive(head,None)
        return self.iter2(head)

    def recursive(self,head,newHead):
        if head == None: return newHead
        nd = head.next
        head.next = newHead
        return self.recursive(nd,head)

    def iter2(self,head):
        prev = None
        while head:
            nd = head.next
            head.next = prev
            prev = head
            head = nd
        return prev

    def iterative(self,head):
        dumy = ListNode(None)
        dumy .next = head
        prev = dumy
        curr = head
        while curr and curr.next:
            nd = curr.next.next
            curr.next.next = prev.next
            prev.next = curr.next
            curr.next = nd
        return dumy.next
        
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#208 Implement Trie (Prefix Tree)
class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.childs = {}
        self.isWord = False


class Trie(object):

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        p = self.root
        for x in word:
            if x not in p.childs:
                child = TrieNode()
                p.childs[x] = child
            p = p.childs[x]
        p.isWord = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.root
        for x in word:
            p = p.childs.get(x)
            if p == None:
                return False
        return p.isWord

    def delete(self, word):
        node = self.root
        queue = [] #this is actually a stack
        for letter in word:
            queue.append((letter, node))
            child = node.childs.get(letter)
            if child is None:
                return False
            node = child
        # no such word in the trie
        if not node.isWord:   return False
        if len(node.childs): # this path has other owrd
            node.isWord = False
        else:
            while queue:
                tmp = queue.pop()
                letter = tmp[0]
                node = tmp[1]
                del node.childs[letter]
                if len(node.childs) or node.isWord:
                    break
        return True

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        p = self.root
        for x in prefix:
            if x in p.childs:
                p = p.childs[x]
            else: return False
        return True

#====dict method====================================
def make_trie(*words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            #setdefault is similar to get(),
            #but will set dict[key]=default if key is not already in dict
            current_dict = current_dict.setdefault(letter, {})
        current_dict['_end_'] = '_end_'
    return root

trie = make_trie('foo', 'bar', 'baz', 'barz')
print trie

def in_trie(word, trie):
    current_dict = trie
    for letter in word:
        if letter not in current_dict:
            return False
        else:
            current_dict = current_dict[letter]
    return '_end_' in current_dict

print in_trie('ba', trie)

#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#----Hard-------------------------------
#214. Shortest Palindrome
"""
Given a string S, you are allowed to convert it to a palindrome by adding characters in front of it.
Find and return the shortest palindrome you can find by performing this transformation. 
For example:

Given "aacecaaa", return "aaacecaaa".

Given "abcd", return "dcbabcd"
"""
class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        return self.sol2(s)

    def sol2(self,s):
        #Rabin-Karp rolling hash
        """
        https://leetcode.com/discuss/95425/8-line-o-n-method-using-rabin-karp-rolling-hash
        """
        n = len(s) ; pos = -1
        B =29; MOD =1000000007; POW = 1;hash1 = 0; hash2 = 0
        for i in  xrange(n):
            hash1 = (hash1 * B + ord(s[i])) % MOD;
            hash2 = (hash2 + ord(s[i]) * POW) % MOD;
            if (hash1 == hash2): pos = i           
            POW = POW *B % MOD
        rev_s =s[pos+1:][::-1]
        return rev_s+s
        
    def sol1(self,s):
        #KMP method对字符串l执行KMP算法，可以得到“部分匹配数组”p（也称“失败函数”）

        #我们只关心p数组的最后一个值p[-1]，因为它表明了rev_s与s相互匹配的最大前缀长度。

        #最后只需要将rev_s的前k个字符与原始串s拼接即可，其中k是s的长度len(s)与p[-1]之差。
        
        rev_s = s[::-1]
        l = s+"#"+rev_s
        p = [0]*len(l)
        for i in xrange(1,len(l)):
            j = p[i-1]
            while j > 0 and l[i]!=l[j]:
                j = p[j-1]
            p[i] = j+(l[i]==l[j])
        return rev_s[:len(s)-p[-1]] +s
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
#-----------------------------------
#216
#-----------------------------------
#217. Contains Duplicate
"""
Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct
"""
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        db ={}
        for num in nums:
            if db.get(num): return True
            db[num] = 1
        return False
        # method2 sort nums 1st o(nlgn), space o(1)
#-----------------------------------
#-----------------------------------
#219. Contains Duplicate II
"""
Given an array of integers and an integer k,
find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k. 
"""
class Solution(object):
    def containsNearbyDuplicate(self, A, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        tb = {}
        for i in xrange(len(A)):
            if A[i] in tb:
                if i-tb[A[i]] <= k : return True
                else: tb[A[i]] = i
            else: tb[A[i]] = i
        return False
#-----------------------------------
#-----------------------------------
#-----------------------------------
#224. Basic Calculator
"""
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .
"""
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        # add white space between elements
        s= re.sub(r'\d+',' \g<0> ',s)
        s= re.sub(r'\(',' \g<0> ',s)
        s= re.sub(r'\)',' \g<0> ',s)
        s= s.split()
        ans = 0
        stack = []
        tempS = []

        for item in s:
            if item.isdigit(): #number
               stack.append(item)
            else:
                if item == ")":
                    while stack[-1] != "(":
                        tempS.append(stack.pop())
                    result = self.onecalculation(tempS)
                    stack.pop() #"("
                    stack.append(result)
                    tempS = []
                else:
                    stack.append(item)

        ans = self.onecalculation(stack,fifo=True)
        return ans

    def onecalculation(self,stack,fifo=False):
        ans = 0
        while stack:
            if fifo: item = stack.pop(0)
            else: item = stack.pop()
            if item == "+":
                item2 = stack.pop(0) if fifo else stack.pop()
                ans += int(item2)
            elif item == "-":
                item2 = stack.pop(0) if fifo else stack.pop()
                ans -= int(item2)
            else: #number
                ans += int(item)
        return ans
#-----------------------------------
#225. Implement Stack using Queues
class Stack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.queue = []
    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.queue.append(x)
    def pop(self):
        """
        :rtype: nothing
        """
        if not self.empty():
            for i in xrange(len(self.queue) -1 ):
                tmp = self.queue.pop(0)
                self.queue.append(tmp)
            self.queue.pop(0)

    def top(self):
        """
        :rtype: int
        """
        if not self.empty():
            for i in xrange(len(self.queue)  ):
                tmp = self.queue.pop(0)
                self.queue.append(tmp)
            return tmp
        else: return None
    def empty(self):
        """
        :rtype: bool
        """
        return self.queue == []
#-----------------------------------
#226. Invert Binary Tree
"""
Invert a binary tree. 
swap nodes in each level
"""
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        #bfs methid
        return self.dfs(root)
        
    def bfs(self,root):
        if root is None: return None
        queue = [root]
        while queue:
            next = queue.pop(0)
            if next.left:
                queue.append(next.left)
            if next.right:
                queue.append(next.right)
            next.left,next.right = next.right,next.left
        return root
        
    def dfs(self,root):
        if root is None: return None
        if root.left: self.dfs(root.left)
        if root.right: self.dfs(root.right)
        root.left,root.right = root.right,root.left
        return root
#----------------------------------
#-----------------------------------
#228. Summary Ranges
"""
Given a sorted integer array without duplicates, return the summary of its ranges.
For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
"""
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        return self.sol2(nums)
        
    def sol2(self,nums):
        x,size = 0,len(nums)
        ans = []
        while x < size:
            c = x
            r = str(nums[x])
            while x +1 <size and nums[x+1]-nums[x] ==1:
                x+=1
            if x>c:
                r += "->"+str(nums[x])
            ans.append(r)
            x+=1
        return ans

    def sol1(self,nums):
        if len(nums) == 0 : return []
        if len(nums) == 1: return [str(nums[0])]
        ret = []
        start = 0
        end = 0
        for i in xrange(1,len(nums)):
            if nums[i] - nums[i-1] == 1: 
                end = i
                continue
            else:
                if end!=start:
                    ret.append(str(nums[start])+"->"+str(nums[end]))
                else:
                    ret.append(str(nums[start]))
                start = i
                end = i
        if end!=start:
            ret.append(str(nums[start])+"->"+str(nums[end]))
        else:
            ret.append(str(nums[start]))    
            
        return ret
#-----------------------------------
#229. Majority Element II
"""
Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times. The algorithm should run in linear time and in O(1) space.
"""
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #at most 2 maj elements
        n = len(nums)
        if n == 0 : return []
        num1 = num2 = None
        count1 = count2 = 0
        for i in xrange(n):
            if count1 == 0:
                num1 = nums[i]
                count1 = 1
            elif count2 == 0 and nums[i]!=num1:
                num2 = nums[i]
                count2 =1
            elif nums[i] == num1: count1+=1
            elif nums[i] == num2: count2+=1
            else:
                count1 -=1
                count2 -=1
        ret = []

        if num1!= None and nums.count(num1) >   n/3 : ret.append(num1)
        if num2!=None and num2 != num1 and  nums.count(num2) >   n/3 : ret.append(num2)
        return ret
#-----------------------------------
#----------------------------------
#-----------------------------------
#232. Implement Queue using Stacks 
"""
 Implement the following operations of a queue using stacks.

    push(x) -- Push element x to the back of queue.
    pop() -- Removes the element from in front of queue.
    peek() -- Get the front element.
    empty() -- Return whether the queue is empt
"""
#push o(n)
#pop o(1)
class Queue(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        tmp = []
        while self.stack:
            tmp.append(self.stack.pop())
        tmp.append(x)
        while tmp:
            self.stack.append(tmp.pop())
    def pop(self):
        """
        :rtype: nothing
        """
        self.stack.pop()
    def peek(self):
        """
        :rtype: int
        """
        return self.stack[-1]
    def empty(self):
        """
        :rtype: bool
        """
        return self.stack == []
#-----------------------------------
#-----------------------------------
#-----------------------------------
#235. Lowest Common Ancestor of a Binary Search Tree
"""
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root.val < p.val and root.val < q.val : return self.lowestCommonAncestor(root.right,p,q)
        elif root.val > p.val and root.val > q.val : return self.lowestCommonAncestor(root.left,p,q)
        else: return root
    def sol2(self, root, p, q):
        while (p.val-root.val)*(q.val-root.val) > 0:
            root = [root.left,root.right][p.val> root.val]
        return root
#-----------------------------------
#236. Lowest Common Ancestor of a Binary Tree
"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        return self.sol1(root,p,q)
    def sol1(self,root,p,q):
        if root == None or root == p or root == q : return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        # left != None means p or q or both in the left tree
        
        if left and right: return root
        elif left and right == None: return left
        elif left == None and right: return right
        else: return None
    def sol2(self,root,p,q):
        if root == None or root == p or root == q : return root
        total = self.countMatchesPA(root.left,p,q)
        if total == 1:
            return root
        elif total ==2:
            return self.sol2(root.left,p,q)
        else:
            return self.sol2(root.right,p,q)
        
    def countMatchesPA(self,root,p,q):
        #count the number of nodes that matches either p or q in the left subtree (which we call totalMatches)
        if root == None: return 0
        matches = self.countMatchesPA(root.left,p,q)
        if root == p or root ==q:
            return 1 + matches
        else:
            return matches
#-----------------------------------
#237. Delete Node in a Linked List 
"""
Write a function to delete a node (except the tail) in a singly linked list, given only access to that node. 
Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.
"""
# we have to replace the value of the node we want to delete with the value in the node after it, and then delete the node after it.
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node.next:
            node.val = node.next.val
            node.next = node.next.next
        else:
            node = None

#-----------------------------------
#-----------------------------------
#-----------------------------------
#240. Search a 2D Matrix II
"""
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

    Integers in each row are sorted in ascending from left to right.
    Integers in each column are sorted in ascending from top to bottom.

"""
# start from upper right point
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) ==0: return False
        y = len(matrix[0])-1
        for x in xrange(len(matrix)):
            while y and matrix[x][y] > target:
                y -=1
            if matrix[x][y] == target:
                return True
        return False
#-----------------------------------
#241
#-----------------------------------
#242. Valid Anagram
"""
Given two strings s and t, write a function to determine if t is an anagram of s.
For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false. 
"""
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t)

#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#274. H-Index
"""
 Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

For example, given citations = [3, 0, 6, 1, 5], which means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively. Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, his h-index is 3

"""
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        return self.solution3(citations)

    def solution2(self,citations):
        #extra space methid
        n = len(citations)
        cnt = [ 0 for i in xrange(n+1)]
        for c in citations:
            if c > n:   cnt[n] +=1 # because h index at most is n
            else:       cnt[c] +=1
        
        sums = 0
        for i in xrange(n,0,-1):
            if sums + cnt[i] >= i : return i
            sums += cnt[i]
        return 0

    def solution1(self,citations):
        # sort method o(nlogn)
        citations = sorted(citations,reverse=True)
        h = 0
        for i in xrange(len(citations)):
            if citations[i] >= i+1: h = i+1
            else: break
        return h
        
    def solution3(self,citations):
        # sort method o(nlogn) + binary search
        citations = sorted(citations,reverse=True)
        n = len(citations)
        if n == 0: return 0
        low = 0; high = n-1; mid = 0
        while low+1 < high:
            mid = (low+high)/2
            if citations[mid] >= mid+1:
                low = mid
            else:
                high = mid
        if  citations[high] >= high+1: return high+1
        elif citations[low] >= low+1:          return low+1
        else: return 0
#-----------------------------------
#275. H-Index II
"""
Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize your algorithm? 
"""
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        # key point is return n-low
        n  = len(citations)
        low = 0; high = n-1

        
        while low <= high:
            mid = (low+high)/2
            if citations[mid] < n-mid: #n-mid the length from end to mid include citation[mid]
                low = mid +1
            else:
                high = mid-1
        return n-low

#-----------------------------------
#-----------------------------------
#-----------------------------------
#278. First Bad Version
"""
 You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API. 
"""
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        lo  = 1
        hi = n
        while lo + 1 < hi:
            ce = lo + (hi-lo)/2
            if isBadVersion(ce):
                hi = ce
            else:
                lo = ce
        if isBadVersion(lo):
            return lo
        if isBadVersion(hi):
            return hi
        return -1
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#292. Nim Game
"""
 You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones.

Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.

For example, if there are 4 stones in the heap, then you will never win the game: no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend. 
"""
class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # pattern is win win win loose win win win loose
        n = n%4
        if n == 0 : return False
        else: return True
#-----------------------------------
#-----------------------------------
#-----------------------------------
#297. Serialize and Deserialize Binary Tree
"""
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Codec:
    def serialize(self, root):
        def doit(node):
            if node:
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                vals.append('#')
        vals= []
        doit(root)
        return " ".join(vals)

    def deserialize(self, data):
        def doit():
            val = next(vals)
            if val == "#": return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node
        vals = iter(data.split())
        return doit()

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
#-----------------------------------
#-----------------------------------
#300. Longest Increasing Subsequence
"""
 Given an unsorted array of integers, find the length of longest increasing subsequence.

For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.

Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity? 
"""
class Solution(object):
    def lengthOfLIS(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        #http://bookshadow.com/weblog/2015/11/03/leetcode-longest-increasing-subsequence/
        #return self.sol1(A)
        return self.sol2(A)

    def sol1(self,A):
        #dp[x] = max(dp[x], dp[y] + 1) 其中 y < x
        size = len(A)
        dp = [1] * size
        for x in xrange(size):
            for y in xrange(x):
                if A[x] > A[y]:
                    dp[x] = max(dp[x],dp[y]+1)
        if dp!=[]: return max(dp)
        else: return 0
    
    def sol2(self,A):
        #binary search method
        #维护一个单调序列
        #遍历nums数组，二分查找每一个数在单调序列中的位置，然后替换之。    
        n= len(A)
        res = []
        for x in xrange(n):
            low,high = 0,len(res)-1
            while low+1 < high:
                mid = (low+high)/2
                if res[mid] < A[x]:
                    low = mid
                else:
                    high = mid
            if low >= len(res):
                res.append(A[x])
            else:
                res[low] = A[x]
        return len(res)
#-----------------------------------
#-----------------------------------
#303. Range Sum Query - Immutable 
"""
Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3

Note:

    You may assume that the array does not change.
    There are many calls to sumRange function.

"""
class NumArray(object):
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        n = len(nums)
        self.sums = [ 0 for i in xrange(n+1)]
        for i in xrange(n):
            self.sums[i+1] = nums[i] + self.sums[i]
        

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sums[j+1] - self.sums[i]
#-----------------------------------
#304. Range Sum Query 2D - Immutable
"""
Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
Note:

    You may assume that the matrix does not change.
    There are many calls to sumRegion function.
    You may assume that row1 ≤ row2 and col1 ≤ col2.

"""
class NumMatrix(object):
    def __init__(self, matrix):
        """
        initialize your data structure here.
        :type matrix: List[List[int]]
        """
        m = len(matrix)

        n = len(matrix[0]) if m else 0
        
        self.sums = [[ 0 for i in xrange(n+1)] for j in xrange(m+1)]
        for j in xrange(1,m+1):
            rowsum = 0
            for i in xrange(1,n+1):
                self.sums[j][i] += rowsum + matrix[j-1][i-1]
                if j >1:
                    self.sums[j][i] +=  self.sums[j-1][i]
                rowsum += matrix[j-1][i-1]
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        sum of elements matrix[(row1,col1)..(row2,col2)], inclusive.
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        # sumRange(row1, col1, row2, col2) = sums[row2][col2] + sums[row1 - 1][col1 - 1] - sums[row1 - 1][col2] - sums[row2][col1 - 1]
        
        return self.sums[row2 + 1][col2 + 1] + self.sums[row1][col1] \
                 - self.sums[row1][col2 + 1] - self.sums[row2 + 1][col1]

#-----------------------------------
#-----------------------------------
#-----------------------------------
#307. Range Sum Query - Mutable
""""
Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
The update(i, val) function modifies nums by updating the element at index i to val. 
Example:
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
"""
class NumArray(object):
    """
    https://leetcode.com/articles/range-sum-query-mutable/
    using segment tree
    http://bookshadow.com/weblog/2015/11/18/leetcode-range-sum-query-mutable/
    """
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        self.nums = nums
        self.size = size = len(nums)
        h = int(math.ceil(math.log(size, 2))) if size else 0
        #the orignal array nums is just the last layer leaves
        maxSize = 2 ** (h + 1) - 1
        self.st = [0] * maxSize
        if size:
            self.initST(0, size - 1, 0)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        if i < 0 or i >= self.size:
            return
        diff = val - self.nums[i]
        self.nums[i] = val
        self.updateST(0, self.size - 1, i, diff, 0)

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        if i < 0 or j < 0 or i >= self.size or j >= self.size:
            return 0
        return self.sumRangeST(0, self.size - 1, i, j, 0)

    def initST(self, ss, se, si):
        """
        si is the index in segment tree
        The segment tree for array a[0,1,…,n−1]a[0, 1, \ldots ,n-1]a[0,1,…,n−1] is a binary tree in which each node contains aggregate information (min, max, sum, etc.) for a subrange [i…j][i \ldots j][i…j] of the array, as its left and right child hold information for range [i…i+j2][i \ldots \frac{i+j}{2}][i…​2​​i+j​​] and [i+j2+1,j][\frac{i + j}{2} + 1, j][​2​​i+j​​+1,j].
        Segment tree could be implemented using either an array or a tree. For an array implementation, if the element at index iii is not a leaf, its left and right child are stored at index 2i2i2i and 2i+12i + 12i+1 respectively.
        """
        if ss == se:
            self.st[si] = self.nums[ss]
        else:
            mid = (ss + se) / 2
            self.st[si] = self.initST(ss, mid, si * 2 + 1) + \
                          self.initST(mid + 1, se, si * 2 + 2)
        return self.st[si]

    def updateST(self, ss, se, i, diff, si):
        if i < ss or i > se:
            return
        # update the node from top root to down leaf
        self.st[si] += diff
        if ss != se:
            mid = (ss + se) / 2
            self.updateST(ss, mid, i, diff, si * 2 + 1)
            self.updateST(mid + 1, se, i, diff, si * 2 + 2)

    def sumRangeST(self, ss, se, qs, qe, si):
        if qs <= ss and qe >= se:
            return self.st[si]
        if se < qs or ss > qe:
            return 0
        mid = (ss + se) / 2
        return self.sumRangeST(ss, mid, qs, qe, si * 2 + 1) + \
                self.sumRangeST(mid + 1, se, qs, qe, si * 2 + 2)

# Your NumArray object will be instantiated and called as such:
# numArray = NumArray(nums)
# numArray.sumRange(0, 1)
# numArray.update(1, 10)
# numArray.sumRange(1, 2)
#-----------------------------------
#319. Bulb Switcher
"""
There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds. 
Example:
Given n = 3. 

At first, the three bulbs are [off, off, off].
After first round, the three bulbs are [on, on, on].
After second round, the three bulbs are [on, off, on].
After third round, the three bulbs are [on, off, off]. 

So you should return 1, because there is only one bulb is on.
"""
class Solution(object):
    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        #http://bookshadow.com/weblog/2015/12/19/leetcode-bulb-switcher/
        # 找有多少完全平方数小于等于n
        """
        为什么只有完全平方数的因子个数为奇数呢？
        因为除了完全平方数，其余数字的因子都是成对出现的，而完全平方数的平方根只会统计一次。
        """
        return int(math.sqrt(n))
#-----------------------------------
#-----------------------------------
#322. Coin Change
"""
You are given coins of different denominations and a total amount of money amount.
Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1. 
"""
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        #dp method
        # dp[x], number of coins to reach amount x
        dp = [-1 for i in xrange(amount+1)]
        dp[0] = 0
        for x in range(amount):
            if dp[x] <0:continue
            for c in coins:
                if x+c > amount: continue
                if dp[x+c] <0 or dp[x+c] > dp[x]+1:
                    dp[x+c] = dp[x]+1
        return dp[amount]
#### if we want to find out how many ways and what are the ways, then use dfs#
# This is same to Combination Sum I
def coin_change(value):
    res = [0, 0, 0]                     # [num_5, num_3, num_1]
    ret = []
    coin_change_helper(0, value, res, ret)
    return ret

def coin_change_helper(cur_face_value, rest_value, res, ret):
    if rest_value == 0:
        ret.append(res[:]) #save a copy since res could change

    for i in range(cur_face_value, 3):
        if rest_value - [5, 3, 1][i] < 0:
            continue
        res[i] += 1
        coin_change_helper(i, rest_value - [5, 3, 1][i], res, ret)
        res[i] -= 1 #very important step
#-----------------------------------
#-----------------------------------
#326. Power of Three
"""
Given an integer, write a function to determine if it is a power of three.
Follow up:
Could you do it without using any loop / recursion? 
"""
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return self.sol2(n)
        
    def sol1(self,n):
        return n>0 and 3** round(math.log(n,3)) == n
    
    def sol2(self,n):
        if n==1: return True
        if n==0 or n%3 > 0 : return False
        return self.sol2(n/3)
#-----------------------------------
#-----------------------------------
#328. Odd Even Linked List
"""
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.
 Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.

Note:
The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ... 

"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None: return head
        odd = oddHead = head
        even = evenHead = head.next
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead
        return oddHead
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#331. Verify Preorder Serialization of a Binary Tree
"""
One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as #
For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character '#' representing null pointer.

You may assume that the input format is always valid, for example it could never contain two consecutive commas such as "1,,3".
"""
class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        """
        num of "#" == n+1
        meet # then pop a number
        """
        A = preorder.split(',')
        n = len(A)
        stack = []
        for i in xrange(n):
            if A[i].isdigit():
                stack.append(A[i])
            else: # "#"
                if stack == []:
                    if i == n-1: return True
                    else: return False
                else:
                    stack.pop()
        return False
#-----------------------------------
#332. Reconstruct Itinerary
"""
Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

    If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
    All airports are represented by three capital letters (IATA code).
    You may assume all tickets form at least one valid itinerary.

Example 1:
tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Return ["JFK", "MUC", "LHR", "SFO", "SJC"].

Example 2:
tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Return ["JFK","ATL","JFK","SFO","ATL","SFO"].
Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"]. But it is larger in lexical order. 
"""
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        """

        http://bookshadow.com/weblog/2016/02/05/leetcode-reconstruct-itinerary/
        """
        return self.sol2(tickets)
    def sol2(self,tickets):
        """
        https://leetcode.com/discuss/84659/short-ruby-python-java-c
        题目的实质就是从JFK顶点出发寻找欧拉通路，可以利用Hierholzer算法。
        """
        targets = collections.defaultdict(list)
        for a, b in sorted(tickets)[::-1]:
            targets[a] += b,
        route = []
        def visit(airport):
            while targets[airport]:
                visit(targets[airport].pop())
            route.append(airport)
        visit('JFK')
        return route[::-1]
        
    def sol1(self,tickets):
        """
        从出发机场开始，按照到达机场的字典序递归搜索
        在搜索过程中删除已经访问过的机票
        将到达机场分为两类：子行程中包含出发机场的记为left，不含出发机场的记为right
        返回时left排在right之前，
        关键点sorted(route[start])不断被调用
        """
        routes = collections.defaultdict(list)
        for s, e in tickets:
            routes[s].append(e)
        for key in routes.keys():
            routes[key]=sorted(routes[key])
            
            
        def solve(start):
            left,right = [],[]
            # left is the subrouties which go back to start, so left should be append first
            for end in sorted(routes[start]):
                if end not in routes[start]:
                    continue
                routes[start].remove(end)
                subroutes = solve(end)
                if start in subroutes:
                    left += subroutes
                else:
                    right += subroutes
            return [start] + left + right
        return solve("JFK")        
#-----------------------------------
#-----------------------------------