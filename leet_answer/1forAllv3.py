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
#2 Add Two numbers
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
#4. Median of Two Sorted Arrays
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
        if n == 1: return s
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
        if n == 1: return s
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
                        dp[i][j] = dp[i][j-2] or dp[i-1][j]  #delete x*to match or the * is used for repeate
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
class Solution(object):
    def threeSum(self, A):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(A) < 3: return []
        res = set([])
        A.sort()
        length = len(A)
        for i in xrange(length-2):
            j = i +1
            k = length-1
            while (j < k):
                if A[j]+A[k] == -1*A[i]:
                    res.add((A[i],A[j],A[k]))
                    j+=1
                    k-=1
                elif A[j]+A[k] < -1*A[i]:
                    j = j+1
                else:
                    k = k-1
        return [list(x) for x in res]
                    
#-----------------------------------
#16. 3Sum Closest
"""
Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

    For example, given array S = {-1 2 1 -4}, and target = 1.

    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

"""
class Solution(object):
    def threeSumClosest(self, A, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        if len(A) < 3: return None
        res = A[0]+A[1]+A[2]
        A.sort()
        length = len(A)
        for i in xrange(length-2):
            j = i +1
            k = length-1
            while (j < k):
                sum = A[j]+A[k] + A[i]
                if sum == target : return sum
                if abs(sum-target)<abs(res-target):
                    res = sum
                if sum < target: j+=1
                if sum > target: k-=1

        return res
#-----------------------------------
#17. Letter Combinations of a Phone Number
"""
Given a digit string, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below.
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
"""
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == None or digits.strip() == "" : return []
        self.dict = {'2':['a','b','c'],
                '3':['d','e','f'],
                '4':['g','h','i'],
                '5':['j','k','l'],
                '6':['m','n','o'],
                '7':['p','q','r','s'],
                '8':['t','u','v'],
                '9':['w','x','y','z']
                }
        self.ret = []
        self.dfs(0,'',digits)
        return self.ret
    def dfs(self,count,vstr,digits):
        if count == len(digits):
            self.ret.append(vstr)
            return
        else:
            for item in self.dict[digits[count]]:
                self.dfs(count+1,vstr+item,digits)

#-----------------------------------
#18. 4Sum
"""
Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

Note: The solution set must not contain duplicate quadruplets.

For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.

A solution set is:
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]

"""
class Solution(object):
    def fourSum(self, A, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(A) < 4: return []
        res = list()
        A.sort()
        length = len(A)
        tb = {} # key 2sum, value index
        #generate all pairs
        for i in xrange(length-1):
            for j in xrange(i+1,length):
                tmp = A[i]+A[j]
                if tmp in tb:
                    tb[tmp].append((i,j))
                else:
                    tb[tmp] = [(i,j)]

        for i in xrange(length-3):
            for j in xrange(i+1,length-2):
                tmp = target - A[i]-A[j]
                if tmp in tb:
                    for item in tb[tmp]:
                        if item[0] > j:
                            res.append([A[i],A[j],A[item[0]],A[item[1]]])
        return [list(x) for x in set([tuple(y)for y in res])]
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
#20. Valid Parentheses
"""
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
"""
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        tb = { '(' : ')',
                '{':'}',
                '[':']'
             }
        tb2 = { ')' : '(',
                '}':'{',
                ']':'['
             }
        stack = []
        for x in s:
            if x in tb:
                stack.append(x)
            else:
                if stack == [] or stack[-1]!= tb2[x]:
                    return False
                stack.pop()
        if stack != [] : return False
        return True
#-----------------------------------
#-----------------------------------
#22. Generate Parentheses
"""
 Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
"""
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        # anytime len(left) >= len (right)
        self.ret = []
        self.dfs(0,0,'',n)
        return self.ret

    def dfs(self,lcount,rcount,vstr,n):
        if lcount == n and rcount == n:
            self.ret.append(vstr)
        if lcount < rcount:
            return
        if lcount < n:
            self.dfs(lcount+1,rcount,vstr+'(',n)
        if rcount < n:
            self.dfs(lcount,rcount+1,vstr+')',n)
                    
#-----------------------------------
#23. Merge k Sorted Lists
"""
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity. 
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """ode]
        :rtype: ListNode
        :type lists: List[ListN
        """
        heap =[]
        for node in lists:
            if node:
                heap.append((node.val,node))
        
        heapq.heapify(heap)
        dummy = ListNode(None)
        curr = dummy
        while heap:
            top = heapq.heappop(heap)
            tmp = ListNode(top[0])
            curr.next = tmp
            curr = curr.next
            if top[1].next:
                heapq.heappush(heap,(top[1].next.val,top[1].next))
        return dummy.next
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
#31. Next Permutation
"""
 Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place, do not allocate extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
"""
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #https://discuss.leetcode.com/topic/52275/easy-python-solution-based-on-lexicographical-permutation-algorithm

        # step1: find the longest non increaseing suffix
        #find nums[i] < nums[i + 1], Loop backwards
        i = -1
        for t in xrange(len(nums) - 2, -1, -1):
            if nums[t] < nums[t+1]:  
                i=t
                break
        # i is the pivot
        if i!= -1:
            # step2: find the rightmost successor to pivot in the suffic
            #find nums[i] < nums[j], Loop backwards
            for j in xrange(len(nums) - 1, i, -1):
                if nums[i] < nums[j]: 
                    # step3: swap betwenn nums[i] and nums[j]
                    nums[i], nums[j] = nums[j], nums[i]
                    break
        
        # step4: reverse the suffix after pivot which is[i + 1, n - 1]
        # revere the nums after pindex
        lo = i+1
        hi = len(nums)-1
        while lo < hi:
            nums[lo],nums[hi] = nums[hi],nums[lo]
            lo +=1
            hi -=1
#-----------------------------------
#32. Longest Valid Parentheses
"""
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

For "(()", the longest valid parentheses substring is "()", which has length = 2.

Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4
"""
class Solution:
    # @param {string} s
    # @return {integer}
    def longestValidParentheses(self, s):
        #Use a stack to record left paren, right paren and index.
        #If current paren is ')' and stack top is '(' then pop up and update maxLen
        stack, maxLen = [-1], 0
        for i in xrange(len(s)):
            if s[i] == ')' and stack[-1] != -1 and s[stack[-1]] == '(':
                stack.pop()
                maxLen = max(maxLen, i - stack[-1])
            else:
                stack.append(i)
        return maxLen
#-----------------------------------
#33. Search in Rotated Sorted Array
"""
Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.
"""
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) ==0 : return None
        lo = 0
        hi = len(nums)-1
        while lo +1  < hi:
            ce = lo + (hi-lo)/2
            if nums[ce] == target : return ce
            elif nums[lo] < nums[ce]:
                #situation 1, numbers between start and mid are sorted
                if target < nums[ce] and target >= nums[lo]:
                    hi = ce
                else:
                    lo = ce
            else:
                #// situation 2, numbers between mid and end are sorted
                if target > nums[ce] and target <= nums [hi]:
                    lo = ce
                else:
                    hi = ce
        if nums[lo] == target : return lo
        elif nums[hi] == target : return hi
        return -1  

#-----------------------------------
#-----------------------------------
#36. Valid Sudoku
"""
Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.

The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
A valid Sudoku board (partially filled) is not necessarily solvable. Only the filled cells need to be validated. 
"""
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # create 3 hashtables
        row = [[False for i in xrange(9)] for j in xrange(9)] #[row][num]
        col = [[False for i in xrange(9)] for j in xrange(9)] #[col][num]
        grid =  [[False for i in xrange(9)] for j in xrange(9)] # (row/3)*3 + j/3
        for x in xrange(9):
            for y in xrange(9):
                if board[x][y] == "." : continue
                num = int(board[x][y]) -1
                if row[x][num] or col[y][num] or grid[(x/3)*3+ (y/3)][num]:
                    return False #duplication found
                else:
                    row[x][num] , col[y][num] , grid[(x/3)*3+ (y/3)][num] = True,True,True
        return True
#-----------------------------------
#37. Sudoku Solver
"""
Write a program to solve a Sudoku puzzle by filling the empty cells.

Empty cells are indicated by the character '.'.

You may assume that there will be only one unique solution. 
"""

class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if len(board) == 0: return False
        self.dfs(board)
    def dfs(self,board):
        for i in xrange(9):
            for j in xrange(9):
                if board[i][j] == ".":
                    for c in xrange(1,10):
                        if self.isvalid(board,i,j,str(c)):
                            board[i][j] = str(c)
                            if self.dfs(board): return True
                            else:
                                board[i][j]= '.'
                    return False
        return True

    def isvalid(self,board, x,y,tmp):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # create 3 hashtables

        for i in xrange(9):
            if board[i][y]==tmp:return False
        for i in xrange(9):
            if board[x][i]==tmp:return False
        for i in xrange(3):
            for j in range(3):
                if board[(x/3)*3+i][(y/3)*3+j]==tmp: return False
        return True 
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
#44. Wildcard Matching
"""
Implement wildcard pattern matching with support for '?' and '*'.
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "*") → true
isMatch("aa", "a*") → true
isMatch("ab", "?*") → true
isMatch("aab", "c*a*b") → false
"""
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        #return self.greedy(s,p)
        return self.dpmethod(s,p)
        
    def dpmethod(self,s,p):
        #Let d(i, j) = true if s[0, i - 1] matches p[0, j - 1] (i, j are string lengths).
        #Initialize:
        #d(0, 0) = true, 
        #d(0, j): if p[j - 1] == '*', d(j) = d(0, j - 1) // deletion; else d(j) = false.
        #Fill up the table:
        #if         p[j - 1] matches s[i - 1],   d(i, j) = d(i - 1, j - 1);
        #else if  p[j - 1] == '*',  find if there is a s[0, k - 1] that matches p[0, j - 1]
        #                                     for (k : 0 to i) { if d(k, j - 1) == true,  d(i, j) = true; }
        #Note: “p[j] matches s[i]” means p[j] == s[i] || p[j] == '?'.  
        # deal with exceeding time limit case
        count = 0
        ls = len(s); lp = len(p)
        for i in xrange(lp):
            if p[i]!="*": count+=1
        if count > ls : return False
        
        dp = [[False for j in range(lp + 1)] for i in range(ls + 1)]  
        
        dp[0][0] = True  
        for j in range(1,len(p) + 1):  
                if p[j-1] == '*':  
                        dp[0][j] = dp[0][j - 1]  
        for i in xrange(1,ls+1):
            for j in xrange(1,lp+1):
                if p[j - 1] == '?' or s[i-1] == p[j-1]:   ## first case   
                    dp[i][j] = dp[i - 1][j - 1] 
                elif p[j-1] == "*" :
                    for k in xrange(i+1):
                        if dp[k][j-1] == True: #that's because if p[0,j-2] can match s[0,k-1] then the * in p can  match the rest in p 
                            dp[i][j] = True
                            break
        return dp[-1][-1]
        
    def greedy(self,s,p):
        #中心思想 *什么都不匹配，你们先比
        # 不行的话，*占一位，然后你们再比
        # 再不行， *占两位
        pPointer = sPointer = ss = 0
        star = -1
        #ss is used to save the place in s when * happened in p
        while sPointer < len(s):
            if pPointer < len(p):
                if p[pPointer] == s[sPointer] or p[pPointer] == '?':
                    pPointer +=1
                    sPointer +=1
                    continue
                elif p[pPointer] == "*" :
                    star = pPointer
                    ss = sPointer
                    pPointer +=1
                    continue
            if star != -1: # e.g. case single '*"
                pPointer = star +1 #always star +1, we try to match this digit in s
                ss +=1
                sPointer = ss
                continue
            return False
        while pPointer < len(p) and p[pPointer] == "*":
            pPointer +=1
        return pPointer == len(p)
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
#46. Permutations
"""
 Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:

[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

"""
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        return self.sol2(nums)

    def sol2(self,nums):
        #iterative use the next permutation method
        if nums is None:  return [[]]
        elif len(nums) <= 1:   return [nums]

        # sort nums first
        nums.sort()

        result = []
        while True:
            result.append([]+nums)
            # step1: find nums[i] < nums[i + 1], Loop backwards
            i = 0
            for i in xrange(len(nums) - 2, -1, -1):
                if nums[i] < nums[i + 1]:   break
                elif i == 0:  return result
            # step2: find nums[i] < nums[j], Loop backwards
            j = 0
            for j in xrange(len(nums) - 1, i, -1):
                if nums[i] < nums[j]:  break
            # step3: swap betwenn nums[i] and nums[j]
            nums[i], nums[j] = nums[j], nums[i]
            # step4: reverse between [i + 1, n - 1]
            nums[i + 1:len(nums)] = nums[len(nums) - 1:i:-1]
        return result

    def sol1(self,nums):
        #recursive
        if len(nums) == 0: return[]
        elif len(nums) ==1: return [nums]
        res = []
        for i in xrange(len(nums)):
            for j in self.sol1(nums[:i]+nums[i+1:]):
                res.append([nums[i]]+j)
        return res
#-----------------------------------
#47. Permutations II
"""
 Given a collection of numbers that might contain duplicates, return all possible unique permutations.

For example,
[1,1,2] have the following unique permutations:

[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

"""
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #
        return self.iterative(nums)

    def iterative(self,nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #https://discuss.leetcode.com/topic/52275/easy-python-solution-based-on-lexicographical-permutation-algorithm


        if nums is None:  return [[]]
        elif len(nums) <= 1:   return [nums]

        # sort nums first
        nums.sort()

        result = set([])
        while True:
            result.add(tuple([]+nums))
            # step1: find nums[i] < nums[i + 1], Loop backwards
            i = 0
            for i in xrange(len(nums) - 2, -1, -1):
                if nums[i] < nums[i + 1]:   break
                elif i == 0:  return [list(x) for x in result]
            # step2: find nums[i] < nums[j], Loop backwards
            j = 0
            for j in xrange(len(nums) - 1, i, -1):
                if nums[i] < nums[j]:  break
            # step3: swap betwenn nums[i] and nums[j]
            nums[i], nums[j] = nums[j], nums[i]
            # step4: reverse between [i + 1, n - 1]
            nums[i + 1:len(nums)] = nums[len(nums) - 1:i:-1]
        return  [list(x) for x in result]

    def recursive(self,nums):
        nums.sort()
        if len(nums) == 0 : return []
        if len(nums) == 1: return [nums]
        res = []
        preNum = None
        for i in xrange(len(nums)):
            if nums[i] == preNum : continue #this step eliminate duplication
            preNum = nums[i]
            for j in self.recursive(nums[:i]+nums[i+1:]):
                res.append([nums[i]]+j)
        return res
#-----------------------------------
#49. Group Anagrams
"""
Given an array of strings, group anagrams together.

For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
Return:

[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]
"""
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        res = []
        tb = {}
        for item in strs:
            key = "".join(sorted(item))
            if key in tb: tb[key].append(item)
            else: tb[key]=[item]
        for key in tb:
            res.append(sorted(tb[key]))
        return res
#-----------------------------------
#50. Pow(x, n)
"""
Implement pow(x, n).
"""
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0: return 1
        elif n < 0 : return 1/self.myPow(x,-n)
        elif n%2: return x*self.myPow(x*x,n/2)
        else: return self.myPow(x*x,n/2)
#-----------------------------------
#53. Maximum Subarray
"""
 Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
the contiguous subarray [4,−1,2,1] has the largest sum = 6. 
"""
class Solution(object):
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #greedy method O(n)
        #return self.greedy(nums)
        # divide and conquer method
        if len(nums)== 0: return None
        
        #return self.divide(nums,0,len(nums)-1)
        
        #dp method
        return self.dp(nums)
        
    def dp(self,nums):
        minsum = Sum = 0
        ret =  -float('inf')
        for i in xrange(len(nums)):
            minsum = min(minsum,Sum)
            Sum += nums[i]
            ret = max(ret,Sum-minsum)
        return ret

    def divide(self,nums,start,end):
        if nums[start] == nums[end]: return nums[start]
        mid = (start+end)/2
        leftans = self.divide(nums,start,mid)
        rightans = self.divide(nums,mid+1,end)
        leftmax = nums[mid]
        rightmax = nums[mid+1]
        tmp =0
        for i in xrange(mid,start-1,-1):
            tmp += nums[i]
            if tmp > leftmax: leftmax = tmp
        tmp = 0
        for i in xrange(mid+1,end+1):
            tmp += nums[i]
            if tmp > rightmax: rightmax = tmp
        return max(max(leftans, rightans),leftmax+rightmax)
        
    def greedy(self,nums):
        currmax = 0
        ret = -float('inf')
        for i in xrange(len(nums)):
            # drop negative sum
            currmax = max(0,currmax)
            currmax += nums[i]
            ret = max(ret,currmax)
        return ret
        
        
#-----------------------------------
#54. Spiral Matrix
"""
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

For example,
Given the following matrix:

[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]

You should return [1,2,3,6,9,8,7,4,5]. 
"""
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        m = len(matrix)
        if m == 0 : return []
        n = len(matrix[0])
        res = []
        direction = 0 # 0 go right, 1 down,2 left,3 up
        left,up = 0,0
        right = n-1
        down = m-1
        # if up > dwon or left > right then return
        while True:
            if direction == 0:
                for i in xrange(left,right+1):
                    res.append(matrix[up][i])
                up +=1
            elif direction == 1:
                for i in xrange(up,down+1):
                    res.append(matrix[i][right])
                right -=1
            elif direction == 2:
                for i in xrange(right,left-1,-1):
                    res.append(matrix[down][i])
                down -=1
            elif direction == 3:
                for i in xrange(down,up-1,-1):
                    res.append(matrix[i][left])
                left +=1
            if up > down or left > right: 
                return res
            direction = (direction+1)%4
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
#-----------------------------------
#-----------------------------------
#59. Spiral Matrix II
"""
Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

For example,
Given n = 3,
You should return the following matrix:

[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]

"""
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n == 0: return []
        

        matrix = [[0 for i in xrange(n)] for j in xrange(n)]
        res = []
        direction = 0 # 0 go right, 1 down,2 left,3 up
        left,up = 0,0
        right = n-1
        down = n-1
        num =1
        # if up > dwon or left > right then return
        while True:
            if direction == 0:
                for i in xrange(left,right+1):
                    matrix[up][i] = num
                    num += 1
                up +=1
            elif direction == 1:
                for i in xrange(up,down+1):
                    matrix[i][right] = num
                    num += 1
                right -=1
            elif direction == 2:
                for i in xrange(right,left-1,-1):
                    matrix[down][i] = num
                    num += 1
                down -=1
            elif direction == 3:
                for i in xrange(down,up-1,-1):
                    matrix[i][left] = num
                    num += 1
                left +=1
            if up > down or left > right: 
                return matrix
            direction = (direction+1)%4
#-----------------------------------
#60. Permutation Sequence
"""
The set [1,2,3,…,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order,
We get the following sequence (ie, for n = 3):

    "123"
    "132"
    "213"
    "231"
    "312"
    "321"

Given n and k, return the kth permutation sequence.

Note: Given n will be between 1 and 9 inclusive.
"""
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        #  k = i_0 * (n - 1)! + i_1 * (n - 2)! + ... + i_{n - 1} * 0!
        # i here is the index in the array num below
        # e.g. k == 6 = 2* 2!+ 1*1!+ 1*0! and num is [1,2,3]
        # so the string is "321"
        
        # get (n-1)!
        fac =1
        for i in xrange(1,n) : fac *= i
        num = range(1,n+1)
        k -= 1  #adjust the input k from 1-based to 0-based index
        res = ''
        for i in xrange(n-1,-1,-1):
            curr = num[k/fac]
            res += str(curr)
            num.remove(curr) # important!
            if i!=0:
                k %= fac
                fac /= i
        return res
        # check the reverse program
        # http://algorithm.yuanbin.me/zh-hans/exhaustive_search/permutation_index.html
#-----------------------------------
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
#69. Sqrt(x)
"""
Implement int sqrt(int x).

Compute and return the square root of x.
"""
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        return self.sol2(x)
        
    def sol2(self,x):
        #http://www.matrix67.com/blog/archives/361
        # newton method
        t = x
        while t**2 >x:
            t = int(t/2.0+x/(2.0*t))
        return t
        
        
    def sol1(self,x):
        #binary search method
        if x == 0 : return 0
        if x <0 : return None
        lo=1
        hi = x/2 +1
        while lo <= hi:
            center = (lo + hi)/2
            if center **2 == x:
                return center
            elif center **2 < x:
                lo = center +1
            else:
                hi = center-1
        return hi
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
#72 Edit Distance
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

    #-----------not recommend--------------------
    #The time complexity of above solution is exponential. In worst case, we may end up doing O(3m) operations. 
    # A Naive recursive Python program to fin minimum number
    # operations to convert str1 to str2
    def editDistance(str1, str2, m , n):
     
        # If first string is empty, the only option is to
        # insert all characters of second string into first
        if m==0:    return n
     
        # If second string is empty, the only option is to
        # remove all characters of first string
        if n==0:    return m
     
        # If last characters of two strings are same, nothing
        # much to do. Ignore last characters and get count for
        # remaining strings.
        if str1[m-1]==str2[n-1]:
            return editDistance(str1,str2,m-1,n-1)
     
        # If last characters are not same, consider all three
        # operations on last character of first string, recursively
        # compute minimum cost for all three operations and take
        # minimum of three values.
        return 1 + min(editDistance(str1, str2, m, n-1),    # Insert
                       editDistance(str1, str2, m-1, n),    # Remove
                       editDistance(str1, str2, m-1, n-1)    # Replace
                       )
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
#75. Sort Colors
"""
 Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively. 
"""
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) <=1 : return
        p0 = 0; p1 = len(nums)-1; i =0
        while i <= p1:
            if nums[i] ==2 :
                nums[p1],nums[i] = nums[i],nums[p1]
                p1 -=1
            elif nums[i] == 0:
                nums[p0],nums[i] = nums[i],nums[p0]
                p0 +=1
                i+=1
            else:
                i += 1
#-----------------------------------
#81. Search in Rotated Sorted Array II
"""
basic idea: always compare nums[ce] with nums[lo], if equal lo+=1

Follow up for "Search in Rotated Sorted Array" #33
What if duplicates are allowed?

Would this affect the run-time complexity? How and why?

Write a function to determine if a given target is in the array.
"""
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        if len(nums) ==0 : return False
        #if len(nums) ==1: return True if  nums[0] == target else False
        lo = 0
        hi = len(nums)-1
        while lo +1 < hi:

            ce = lo + (hi-lo)/2
            if nums[ce] == target : return True
            if nums[ce] > nums[lo]:
                if target < nums[ce] and target >= nums [lo]:
                    hi = ce
                else:
                    lo = ce

            elif nums[ce] < nums[lo]:
                if target > nums[ce] and target <= nums[hi]:
                    lo = ce 
                else:
                    hi = ce
            else:
                lo +=1
            
        if nums[lo] == target : return True
        elif nums[hi] == target : return True
        return False

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
#84. Largest Rectangle in Histogram
"""
Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram. 
"""
class Solution(object):
    def largestRectangleArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # use stacks to store height and index
        # if height[i] > stack top, push in
        # if height[i] == stack top,ignore
        # if height[i] < stacktop, pop and calculate tmp area untill stack top <= height[i]
        stackH = []
        stackI = [] #index
        maxArea = 0
        for i in xrange(len(height)):
            if stackH == [] or height[i] > stackH[-1]:
                stackH.append(height[i])
                stackI.append(i)
            elif height[i] < stackH[-1]:
                lastIndex = 0
                while stackH and height[i] < stackH[-1]:
                    lastIndex = stackI.pop()
                    tmpArea = stackH.pop() * (i-lastIndex)
                    maxArea = max(maxArea,tmpArea)
                if stackH == [] or stackH[-1] < height[i]:
                    stackH.append(height[i])
                    stackI.append(lastIndex) #very important !! 
        n =len(height) 
        while stackH:
            tmpArea = stackH.pop() * (n -stackI.pop())
            maxArea = max(maxArea,tmpArea)
        return maxArea

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
#94 Binary Tree Inorder Traversal
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
#95. Unique Binary Search Trees II
"""
Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1...n.

For example,
Given n = 3, your program should return all 5 unique BST's shown below.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

"""
class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
    
        def dfs(start,end):
            if start > end : return [None]
            res = []
            for rootval in xrange(start,end+1):
                leftTree = dfs(start,rootval-1)
                rightTree = dfs(rootval+1,end)
                for i in leftTree:
                    for j in rightTree:
                        root = TreeNode(rootval)
                        root.left = i
                        root.right = j
                        res.append(root)
            return res
        return dfs(1,n)

#-----------------------------------
#96. Unique Binary Search Trees
"""
Given n, how many structurally unique BST's (binary search trees) that store values 1...n?

For example,
Given n = 3, there are a total of 5 unique BST's.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

"""
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        return self.mathmethod(n)
        
    def mathmethod(self,n):
        #Catlan number
        c=math.factorial(2*n)/math.factorial(n)
        c /= math.factorial(n+1)
        return c
        
    def numTrees2(self, minV, maxV):
        # dfs method time exceed
        if minV >= maxV:
            return 1
        val = 0
        for i in range(minV,maxV+1):
            val = val + self.numTrees2(minV, i-1)*self.numTrees2(i+1, maxV)
        return val
        
    def dpmethod(self,n):
        # http://algorithm.yuanbin.me/zh-hans/math_and_bit_manipulation/unique_binary_search_trees.html
        dp = [0 for i in xrange(n+1)]
        dp[0] =1
        dp[1]=1
        for i in xrange(2,n+1):
            for j in xrange(0,i):
                dp[i] += dp[j]*dp[i-1-j]
        return dp[n]
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

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        #return self.dfs(root,-1*float('inf'),float('inf'))
        self.prev = None
        return self.sol2(root)

    def dfs(self,root,min,max):
        if root == None: return True
        if root.val <= min or root.val >= max: return False
        return   self.dfs(root.left,min,root.val) and  self.dfs(root.right,root.val,max)
        
    def sol2(self,node):
        """
        If we use in-order traversal to serialize a binary search tree, we can
        get a list of values in ascending order. It can be proved with the
        definition of BST. And here I use the reference of TreeNode
        pointer prev as a global variable to mark the address of previous node in the
        list.
        """
        #inroder method
        if node == None : return True
        if not self.sol2(node.left): return False
        if self.prev!= None and self.prev.val >= node.val : return False
        self.prev = node
        return self.sol2(node.right)
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
#102. Binary Tree Level Order Traversal
"""
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
"""
class Solution(object):
    def levelOrder(self, root):
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
            self.res.append(tmplist)
            if queue: queue.append('end')

    def dfs(self,root,level):
        if root:
            if len(self.res) < level +1:
                self.res.append([])
            self.res[level].append(root.val)
            self.dfs(root.left,level+1)
            self.dfs(root.right,level+1)
#-----------bfs------------------------
#103. Binary Tree Zigzag Level Order Traversal
"""
Given a binary tree, return the zigzag level order traversal of its nodes' values.
(ie, from left to right, then right to left for the next level and alternate between).
"""
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
#105. Construct Binary Tree from Preorder and Inorder Traversal
"""
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder: return None
        self.preorder,self.inorder = preorder, inorder
        return self.dfs(0,0,len(preorder))
    
    def dfs(self,preLeft,inLeft,Len):
        if Len == 0 : return None
        root = TreeNode(self.preorder[preLeft])
        rootPos = self.inorder.index(root.val)  #since no duplication
        root.left = self.dfs(preLeft +1,inLeft,rootPos - inLeft)
        root.right = self.dfs(preLeft +1 +rootPos - inLeft,rootPos+1,Len-1-(rootPos-inLeft) )
        return root
#-----------------------------------
#106. Construct Binary Tree from Inorder and Postorder Traversal
"""
Given inorder and postorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder: return None
        self.postorder,self.inorder = postorder, inorder
        return self.dfs(0,0,len(postorder))
    
    def dfs(self,postLeft,inLeft,Len):
        if Len == 0 : return None
        root = TreeNode(self.postorder[postLeft+Len-1])
        rootPos = self.inorder.index(root.val)  #since no duplication
        root.left = self.dfs(postLeft,inLeft,rootPos - inLeft)
        root.right = self.dfs(postLeft  +rootPos - inLeft,rootPos+1,Len-1-(rootPos-inLeft) )
        return root
#-----------------------------------
#107. Binary Tree Level Order Traversal II
"""
Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
"""
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        self.res = []
        #self.dfs(root,0)
        self.bfs(root)
        self.res.reverse()
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
            self.res.append(tmplist)
            if queue: queue.append('end')

        
    def dfs(self,root,level):
        if root:
            if len(self.res) < level +1:
                self.res.append([])
            self.res[level].append(root.val)
            self.dfs(root.left,level+1)
            self.dfs(root.right,level+1)
#-----------------------------------
#108. Convert Sorted Array to Binary Search Tre
"""
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.dfs(nums,0,len(nums)-1)
        
    def dfs(self,nums,start,end):
        if start > end : return None
        mid = start + (end-start)/2
        root = TreeNode(nums[mid])
        root.left = self.dfs(nums,start,mid-1)
        root.right = self.dfs(nums,mid +1 ,end)
        return root
#-----------------------------------
#109. Convert Sorted List to Binary Search Tree
"""
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# trick: convert to array first
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        nums = []
        curr = head
        while curr:
            nums.append(curr.val)
            curr = curr.next
        return self.dfs(nums,0,len(nums)-1)
        
    def dfs(self,nums,start,end):
        if start > end : return None
        mid = (start + end)/2
        root = TreeNode(nums[mid])
        root.left = self.dfs(nums,start,mid-1)
        root.right = self.dfs(nums,mid +1 ,end)
        return root 
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#114 Flatten Binary Tree to Linked List
"""
Given a binary tree, flatten it to a linked list in-place. 
 For example,
Given

         1
        / \
       2   5
      / \   \
     3   4   6

The flattened tree should look like:

   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6

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
        # set curr.right to curr.left
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
#121. Best Time to Buy and Sell Stock
"""
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Example 1:

Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)

Example 2:

Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.

"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)  == 0 : return 0
        minv = prices[0]
        ret = 0
        for i in xrange(len(prices)):
            minv = min(minv,prices[i])
            ret = max(ret,prices[i]-minv)
        return ret
#-----------------------------------
#122. Best Time to Buy and Sell Stock II
"""
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0 : return 0
        ret = 0
        for i in xrange(1, len(prices)):
            if prices[i] > prices[i-1]: ret += prices[i] - prices[i-1]
        return ret
#-----------------------------------
#123. Best Time to Buy and Sell Stock III
"""
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.
"""
class Solution(object):
    def maxProfit(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        
        lenA = len(A)
        if lenA < 2 : return 0
        f = [0 for i in xrange(lenA)]
        g = [0 for i in xrange(lenA)]

        minv = A[0]
        for i in xrange(1,lenA):
            minv = min(minv,A[i])
            f[i]  = max(f[i-1],A[i]-minv)
        
        maxv = A[lenA-1]
        for j in xrange(lenA-2,-1,-1):
            maxv = max(maxv,A[j])
            g[j] = max(g[j+1],maxv-A[j])
            
        ret = 0
        for i in xrange(lenA):
            ret = max(ret,f[i]+g[i])
        return ret
#-----------------------------------
#125. Valid Palindrome
"""
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome. 
"""
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s == None: return None
        s= filter(lambda x:x.isalnum(),s)
        s= s.lower()
        if len(s)<=1: return True
        p0 =0; p1=len(s)-1
        while p0<p1:
            if s[p0]!=s[p1]:
                return False
            p0+=1
            p1-=1
        return True
#-----------------------------------
#126. Word Ladder II
"""
 Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

    Only one letter can be changed at a time
    Each intermediate word must exist in the word list

For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Return

  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]

Note:

    All words have the same length.
    All words contain only lowercase alphabetic characters.

"""
class Solution(object):
    def findLadders(self, start, end, wordlist):
        """
        :type beginWord: str
        :type endWord: str
        :type wordlist: Set[str]
        :rtype: List[List[int]]
        """
        # thanks to https://leetcode.com/discuss/24191/defaultdict-for-traceback-and-easy-writing-lines-python-code
        #http://chaoren.is-programmer.com/posts/43039.html
        # use collections.defaultdict(set) which will init each key with valie set() and avoid duplication
        # key is childword, value is a set of parents word
        wordlist.add(end)
        level = set([start])  #it's like a queue for bfs
        # key is word, value is parent word, e.g. {'hot': set(['hit']), 'cog': set(['log', 'dog'])}
        # In each level, defaultdict(set) can remove duplicates, first we need to get parent dictionary
        hashtable = collections.defaultdict(set)
        lw = len(start)
        # dictionary.update(dictionary) function can add or update key-pair
        while level and end not in hashtable: #if end also a key in hashtable means we have already found solution
            nextlevel = collections.defaultdict(set)
            for word in level:
                for i in xrange(lw):
                    part1 = word[:i];part2 = word[i+1:]
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        if j!= word[i]:
                            nextword = part1 + j +part2
                            if nextword in wordlist and nextword not in hashtable:
                                  #!!!tricky part! nextowrd will be added into hasttable untill hashtable.update(nextlevel)
                                  nextlevel[nextword].add(word) 
            level = nextlevel
            hashtable.update(nextlevel)
        res = [[end]]
        while res and res[0][0]!=start:
            res = [[p] + r for r in res for p in hashtable[r[0]]]
        return res   
            
#-----------------------------------
#127. Word Ladder
"""
 Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

    Only one letter can be changed at a time
    Each intermediate word must exist in the word list

For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:

    Return 0 if there is no such transformation sequence.
    All words have the same length.
    All words contain only lowercase alphabetic characters.

"""
class Solution(object):
    def ladderLength(self, start, end, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: Set[str]
        :rtype: int
        """
        #BFS method
        q = []
        q.append((start,1)) #word and length
        wordList.add(end)
        lw = len(start)
        while q:
            curr = q.pop(0)
            currWord = curr[0]
            currLen = curr[1]
            if currWord == end: return currLen
            for i in xrange(lw):
                left = currWord[:i]
                right = currWord[i+1:]
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    if currWord[i]!=j:
                        nextWord = left + j + right
                        if nextWord in wordList:
                            q.append((nextWord,currLen+1))
                            wordList.remove(nextWord)
        return 0
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
#133. Clone Graph
"""
 Clone an undirected graph. Each node in the graph contains a label and a list of its neighbours.

OJ's undirected graph serialization:

Nodes are labeled uniquely.
We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.

As an example, consider the serialized graph {0,1,2#1,2#2,2}.

The graph has a total of three nodes, and therefore contains three parts as separated by #.

    First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
    Second node is labeled as 1. Connect node 1 to node 2.
    Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.

Visually, the graph looks like the following: 
       1
      / \
     /   \
    0 --- 2
         / \
         \_/
"""
# Definition for a undirected graph node
# class UndirectedGraphNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: UndirectedGraphNode
        :rtype: UndirectedGraphNode
        """
        if node == None: return None
        #dfs method
        #return self.dfs(node,{})
        return self.bfs(node)
    
    def bfs(self,node):
        # hashtable key--visited node, value--its copy to avoid duplication
        map = {} # indicate visited
        queue = []
        copy = UndirectedGraphNode(node.label)
        map[node] = copy
        queue.append(node) #element in queue all from the original graph
        while queue:
            v = queue.pop(0)
            for w in v.neighbors:
                #w is one neighbour of v the origin graph
                # map[w] is also a copy of w
                if w in map:
                    map[v].neighbors.append(map[w])
                else:
                    wcopy = UndirectedGraphNode(w.label)
                    map[w] = wcopy
                    map[v].neighbors.append(wcopy)
                    queue.append(w)
        return copy

    def dfs(self,input,map):
        # hash table key--visited node, value--its copy to avoid duplication!!! smart
        if input in map: return map[input]
        output = UndirectedGraphNode(input.label)
        map[input]=output
        for neighbor in input.neighbors:
            output.neighbors.append(self.dfs(neighbor,map))
        return output
#-----------------------------------
#136. Single Number
"""
Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory? 
"""
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ln = len(nums)
        if ln == 0: return None
        res = nums[0]
        if ln == 1: return res
        for i in xrange(1,ln):
            res  ^= nums[i]
        return res
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
#140. Word Break II
"""
 Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

Return all such possible sentences.

For example, given
s = "catsanddog",
dict = ["cat", "cats", "and", "sand", "dog"].

A solution is ["cats and dog", "cat sand dog"]. 
"""
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        return self.sol2(s,wordDict)
    
    def sol2(self,s,wordDict):
        tokenDict = {} #key is string suffix, value is a list how it is splited
        def dfs(s):
            ans = []
            if s in wordDict: ans.append(s)
            for x in xrange(len(s)-1):
                prefix,suffix =  s[:x+1],s[x+1:]
                if prefix not in wordDict: continue
                rest = []
                if suffix in tokenDict: rest = tokenDict[suffix]
                else:                  rest = dfs(suffix)
                for n in rest:
                    ans.append(prefix + ' ' + n)
            tokenDict[s] = ans
            return ans
        
        return dfs(s)
    
    #----------------------    
    def sol1(self, s, wordDict):
        # use the dp method + dfs, not efficient
        self.res = []
        self.dfs(s,wordDict,'')
        return self.res
        
        
    def dfs(self,s,wordDict,vstr):
        if self.check(s,wordDict):
            if len(s) == 0: 
                self.res.append(vstr[1:])
            for i in xrange(1,len(s)+1):
                if s[:i] in wordDict:
                    self.dfs(s[i:],wordDict,vstr+ ' ' + s[:i])
        
    def check(self,s,wordDict):
        n = len(s)
        dp = [False for i in xrange(n+1)]
        dp[0] = True
        for i in xrange(1,n+1):
            for j in xrange(0,i):
                if dp[j] and s[j:i] in wordDict: 
                    dp[i] = True
                    break
        return dp[-1]
        
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
#149. Max Points on a Line
"""
Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
"""
# Definition for a point.
# class Point(object):
#     def __init__(self, a=0, b=0):
#         self.x = a
#         self.y = b

class Solution(object):

    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        res = 0
        lp = len(points)
        if lp < 3 : return lp
        for i in xrange(lp):
            samepoints =1
            slope = {'inf':0}
            for j in xrange(i+1,lp):
                if points[i].x == points[j].x and points[i].y != points[j].y:
                    slope['inf'] += 1
                elif points[i].x != points[j].x:
                    tmp = 1.0 * 1.0 * (points[i].y - points[j].y) / (points[i].x - points[j].x)
                    if tmp in slope : slope[tmp] += 1
                    else:             slope[tmp] = 1
                else: samepoints+=1
            res = max(res,max(slope.values())+samepoints)
        return res

#-----------------------------------
#153. Find Minimum in Rotated Sorted Array
"""
Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.

You may assume no duplicate exists in the array.
"""
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) ==0 : return None
        lo = 0
        hi = len(nums)-1
        while lo +1 < hi:
            ce = (lo+hi)/2
            if nums[ce] > nums[hi]:
                lo = ce
            else:
                hi = ce
        if nums[lo]<nums[hi]: return nums[lo]
        else: return nums[hi]
#-----------------------------------
#154. Find Minimum in Rotated Sorted Array II
"""


    Follow up for "Find Minimum in Rotated Sorted Array":
    What if duplicates are allowed?

    Would this affect the run-time complexity? How and why?

Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.

The array may contain duplicates.

Subscribe to see which companies asked this question

"""
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) ==0 : return None

        lo = 0
        hi = len(nums)-1
        while lo+1 < hi and nums[lo] >= nums[hi]:
            ce = (lo+hi)/2
            if nums[ce] > nums[lo]:
                lo = ce 
            elif nums[ce] < nums[lo]:
                hi = ce
            else:
                lo +=1
        if nums[lo] < nums[hi]: return nums[lo]
        else: return nums[hi]


#-----------------------------------
#166. Fraction to Recurring Decimal

class Solution(object):
    """
    Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

    If the fractional part is repeating, enclose the repeating part in parentheses.

    For example,

        Given numerator = 1, denominator = 2, return "0.5".
        Given numerator = 2, denominator = 1, return "2".
        Given numerator = 2, denominator = 3, return "0.(6)".

    """
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
                # cnt +=1 so that make sure the last character in rl is included
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
#173. Binary Search Tree Iterator
"""
Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree. 
"""
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        # constructor go to the most left node
        self.stack =[]
        self.pushLeft(root)
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.stack != []

    def next(self):
        """
        :rtype: int
        """
        top = self.stack.pop()
        self.pushLeft(top.right)
        return top.val
        
        
    def pushLeft(self,root):
        #push node into stack untill
        # find the most left node
        while root:
            self.stack.append(root)
            root = root.left
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())
#-----------------------------------
#-----------------------------------
#187. Repeated DNA Sequences
"""
 All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.

Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.

For example,

Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",

Return:
["AAAAACCCCC", "CCCCCAAAAA"].

"""
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        return self.sol1(s)
        
    def sol1(self,s):
        #hash table + bitmap
        ans = []
        table = {}
        map = {'A':0,'C':1,'G':2,'T':3} # 00 01 10 11
        sum = 0
        for i in xrange(len(s)):
            sum = ((sum<<2) | map[s[i]]) & 0xFFFFF
            if i < 9: continue
            table[sum] = table.get(sum,0) + 1
            if table[sum] ==2 :
                ans.append(s[i-9:i+1])
        return ans
    

    def sol2(self, s):
        # Robin-Karp method
        """
        s test, P pattern
        """
        ls = len(s)
        lp = 10
        ans = []
        if ls < lp: return ans

        prime = 1000000007#big prime
        x = 29 # random(1,prime-1)
        
        db = {}
        # precomputeHashes
        H = [None for i in range(ls-10+1)]
        S = s[ls-lp:ls]
        H[ls-lp] = self.PolyHash(S,prime,x)
        db[H[ls-lp]] = 1
        
        y =1
        for i in range(1,lp+1):
            y = (y*x) %prime
        for i in range(ls-lp-1,-1,-1):
            H[i] = (x*H[i+1]+ord(s[i])-y*ord(s[i+lp]))%prime
            db[H[i]] = db.get(H[i],0) + 1
            if db[H[i]] == 2:
                ans.append(s[i:i+lp])

        ans.reverse()
        return ans
        
    def PolyHash(self,P,prime,x):
        ans = 0
        for c in reversed(P):
            ans = (ans * x + ord(c)) % prime
        return ans % prime
#-----------------------------------
#188. Best Time to Buy and Sell Stock IV
"""
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most k transactions.
"""
class Solution(object):
    def maxProfit(self, k, A):
        """
        :type k: int
        :type A: List[int]
        :rtype: int
        """
        #http://wlcoding.blogspot.com/2015/03/best-time-to-buy-and-sell-stock.html?view=sidebar
        # try dp size [k][lenA]
        """
        
    There is no sell at j: d(i, j) = d(i, j - 1)
    There is a sell at j (there must be a buy at l < j): d(i, j) = max_{0 <= l < j} {d(i - 1, l - 1) - p[l] + p[j]}

    d(i, j) = max{d(i, j - 1), max_{1 <= l <= j} {d(i - 1, l - 1) - p[l] + p[j]}}
         = max{d(i, j - 1), p[j] + max_{1 <= l <= j} {d(i - 1, l - 1) - p[l]}} // move p[j] out of max
        profit_max = d[k][N - 1]
        """

        lenA = len(A)
        if len(A) < 2 : return 0
        if k > lenA/2 : return self.quickSolve(lenA, A) 
        dp = [[0 for i in xrange(lenA)] for j in xrange(k+1)]
        # d(i, j) = max{d(i, j - 1), max_{1<=l<=j} {d(i - 1, l - 1) - p[l] (buy) + p[j] (sell)}}
        for i in xrange(1,k+1):
            tmpmax = dp[i-1][0]-A[0]
            for j in xrange(1,lenA):
                #find the point although buy stock, the money left is max
                tmpmax = max(tmpmax,dp[i - 1][j - 1] - A[j])
                #dp[i][j-1] means do noting at j
                dp[i][j] = max(dp[i][j-1], tmpmax+ A[j])

        return dp[k][lenA-1]

    def quickSolve(self, size, prices):
        sum = 0
        for x in range(size - 1):
            if prices[x + 1] > prices[x]:
                sum += prices[x + 1] - prices[x]
        return sum 

#-----------------------------------
#189. Rotate Array
"""
Rotate an array of n elements to the right by k steps.
For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4]. 
"""
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        self.reverse(nums, 0, n - k)
        self.reverse(nums, n - k, n)
        self.reverse(nums, 0, n)

    def reverse(self, nums, start, end):
        for x in range(start, (start + end) / 2):
            nums[x] ^= nums[start + end - x - 1]
            nums[start + end - x - 1] ^= nums[x]
            nums[x] ^= nums[start + end - x - 1]
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
#192. Word Frequency
"""
Write a bash script to calculate the frequency of each word in a text file words.txt.

For simplicity sake, you may assume:

    words.txt contains only lowercase characters and space ' ' characters.
    Each word must consist of lowercase characters only.
    Words are separated by one or more whitespace characters.

For example, assume that words.txt has the following content:

the day is sunny the the
the sunny is is

Your script should output the following, sorted by descending frequency:

the 4
is 3
sunny 2
day 1

Note:
Don't worry about handling ties, it is guaranteed that each word's frequency count is unique. 
"""
# Read from the file words.txt and output the word frequency list to stdout.
#http://bookshadow.com/weblog/2015/03/24/leetcode-word-frequency/
#tr -s: 使用指定字符串替换出现一次或者连续出现的目标字符串（把一个或多个连续空格用换行符代替）

#sort: 将单词从小到大排序

#uniq -c: uniq用来对连续出现的行去重，-c参数为计数

#sort -rn: -r 倒序排列， -n 按照数值大小排序（感谢网友 长弓1990 指正）

#awk '{ print $2, $1 }': 格式化输出，将每一行的内容用空格分隔成若干部分，$i为第i个部分。
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -rn | awk '{print $2" "$1}'
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
#200. Number of Islands
"""Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
"""
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # for python, if input variable is array, it is mutable, i.e the value get changed after the function
        # call
        return self.sol1(grid)
    
    def sol1(self,grid):
        self.row = len(grid)
        if self.row ==0: return 0
        self.col = len(grid[0])
        ret = 0
        for i in xrange(self.row):
            for j in xrange(self.col):
                if grid[i][j] == "1":
                    ##dfs solution
                    #self.dfs(grid,i,j)
                    #bfs solution
                    self.bfs(grid,i,j)
                    ret +=1
        # ideally we should recover the grid
        return ret
        
    def dfs(self,grid,i,j):
        if grid[i][j] == "0" : return
        if grid[i][j] == "1":
            grid[i][j] = "#" #do not traverse the same point again
            if i >= 1: self.dfs(grid,i-1,j)
            if i < self.row-1 : self.dfs(grid,i+1,j)
            if j >= 1 : self.dfs(grid,i,j-1)
            if j < self.col-1: self.dfs(grid,i,j+1)
    def bfs(self,grid,i,j):
        queue = [] # each ceil in grid labledd as i*self.col + j
        self.visit(grid,i,j,queue)
        while queue!=[]:
            pos = queue.pop(0)
            row = pos/self.col
            col = pos%self.col
            self.visit(grid,row-1,col,queue)
            self.visit(grid,row+1,col,queue)
            self.visit(grid,row,col-1,queue)
            self.visit(grid,row,col+1,queue)
        
    def visit(self,grid,i,j,queue):
        if i<0 or i > (self.row-1) or j <0 or j > (self.col-1) or grid[i][j]!="1": return
        grid[i][j] = "#"
        queue.append(i*self.col + j)
#-----------------------------------
#-----------------------------------
#202. Happy Number
"""
Write an algorithm to determine if a number is "happy".

A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

Example: 19 is a happy number

    12 + 92 = 82
    82 + 22 = 68
    62 + 82 = 100
    12 + 02 + 02 = 1
"""
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        l= []
        while True:
            sum =0
            while n:
                digit = n%10
                sum += digit**2
                n = n/10
            n = sum
            if n== 1 or (n in l): break
            l.append(n)
        return n==1
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
#207. Course Schedule
"""
 There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

For example:

2, [[1,0]]

There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.

2, [[1,0],[0,1]]

There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
"""
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        #BFS method
        # a->b indegree of b is 1 and degree of a is 0
        # the key is find all root courses and remove them
        
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
        return len(A) == 0
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
#209. Minimum Size Subarray Sum
"""
 Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of which the sum ≥ s. If there isn't one, return 0 instead.

For example, given the array [2,3,1,2,4,3] and s = 7,
the subarray [4,3] has the minimal length under the problem constraint.

click to show more practice.
More practice:

If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n).

"""
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        # 
        #if len(nums) == 0: return 0
        return self.sol2(s,nums)
        
        
    def sol2(self,s,nums):
        # o(nlgn) method
        size = len(nums)
        left,right = 0,size
        ret = 0
        while left <= right:
            mid = (left+right)/2
            if self.solve(mid,s,nums):
                ret = mid
                right = mid-1
            else:
                left = mid +1
        return ret
        
        
    def solve(self,l,s,nums):
        sums = 0
        for x in xrange(len(nums)):
            sums += nums[x]
            if x >= l :
                sums -= nums[x-l]
            if sums >=s :
                return True
        return False
            
    def sol1(self,s,nums):
        #o(N) method
        sum = 0
        ret = None
        p1  = 0
        for i in xrange(len(nums)):
            sum += nums[i]
            while sum >= s :
                ret = i-p1 +1 if ret == None else min(ret, i-p1 +1)
                sum -= nums[p1]
                p1 += 1
                if p1 > len(nums) -1: break
        if ret == None: return 0
        else: return ret
#-----------------------------------
#210. Course Schedule II
"""
 There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

For example:

2, [[1,0]]

There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]

4, [[1,0],[2,0],[3,1],[3,2]]

There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].
"""
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        #BFS method
        # a->b degree of b is 1 and degree of a is 0
        degrees = [0 for i in xrange(numCourses)]
        childs = [[] for i in xrange(numCourses)]
        for pair in prerequisites:
            degrees[pair[0]]+=1
            childs[pair[1]].append(pair[0])
        A = set(range(numCourses)) #courses
        
        delqueue = []
        ans = []
        # find all root course
        for course in A:
            if degrees[course] == 0:
                delqueue.append(course)
        while delqueue:
            course = delqueue.pop(0)
            ans.append(course)
            A.remove(course)
            for child in childs[course]:
                degrees[child]-=1
                if degrees[child] == 0:
                    delqueue.append(child)
        return [[],ans][len(A) == 0]
        
#-----------------------------------
#-----------------------------------
#213. House Robber II
"""
After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
"""
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return 0
        if n <=2 : return max(nums)
        return max(self.helper(nums[:-1]),self.helper(nums[1:]))
    def helper(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)

        odd, even= 0,0
        
        for i in xrange(n):
            if i %2:
                odd = max(odd+nums[i],even)
            else:
                even = max(even+nums[i],odd)
        return max(even,odd)
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
        our goad is to divide s to A+B while A is palindrome, then return revB+A+B
        using Rabin-Karp to check A is palindrome
        """
        n = len(s) ; pos = -1
        B =29; MOD =1000000007; POW = 1;hash1 = 0; hash2 = 0
        for i in  xrange(n):
            hash1 = (hash1 * B + ord(s[i])) % MOD;
            hash2 = (hash2 + ord(s[i]) * POW) % MOD;
            #if (hash1 == hash2) and self.isPalindrome(s[:i+1]):  #Time exceed limit
            if (hash1 == hash2)
                pos = i           
            POW = POW *B % MOD
        rev_s =s[pos+1:][::-1]
        return rev_s+s

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        lens = len(s)
        for i in xrange(lens/2):
            if s[i] != s[lens-1-i]: return False
        return True

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
#220. Contains Duplicate III
"""
Given an array of integers, find out whether there are two distinct indices i and j in the array such that the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k. 
"""
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        # java equivalent treeset in python is collections.OrderedDict()
        # http://bookshadow.com/weblog/2015/06/03/leetcode-contains-duplicate-iii/
        if t < 0 or k <1 : return False
        db = collections.OrderedDict()
        for i in xrange(len(nums)):
            key = nums[i]/max(1,t)
            for m in (key-1,key,key+1):
                if m in db and (abs(nums[i] - db[m]) <=t): return True
            db[key] = nums[i]
            if i >=k:
                db.popitem(last=False)
        return False
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
#234. Palindrome Linked List
"""
Given a singly linked list, determine if it is a palindrome.
Follow up:
Could you do it in O(n) time and O(1) space?
"""
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None : return True
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        p1 = self.reverseList(slow)
        p2 = head
        while p2!= slow and p2.val == p1.val:
            p1 = p1.next
            p2 = p2.next
        return p2 == slow

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
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
#268. Missing Number
"""
 Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

For example,
Given nums = [0, 1, 3] return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity? 
"""
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # using xor to eliminiate all same number
        a=reduce(lambda x,y: x^y, nums + range(len(nums)+1))
        return a

    def sol1(self,nums):
        n = len(nums)
        sumn = sum(nums)
        return n*(n+1)/2 - sumn
#-----------------------------------
#272. Zigzag Iterator
"""
Given two 1d vectors, implement an iterator to return their elements alternately.

For example, given two 1d vectors:

v1 = [1, 2]
v2 = [3, 4, 5, 6]
By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1, 3, 2, 4, 5, 6].

Follow up: What if you are given k 1d vectors? How well can your code be extended to such cases?

Clarification for the follow up question - Update (2015-09-18):
The "Zigzag" order is not clearly defined and is ambiguous for k > 2 cases. If "Zigzag" does not look right to you, replace "Zigzag" with "Cyclic". For example, given the following input:

[1,2,3]
[4,5,6,7]
[8,9]
It should return [1,4,8,2,5,9,3,6,7].
"""
class ZigzagIterator(object):  
  
    def __init__(self, v1, v2):  
        """ 
        Initialize your data structure here. 
        :type v1: List[int] 
        :type v2: List[int] 
        """  
        self.l = []  
        i = 0  
        while i < max(len(v1), len(v2)):  
            if i < len(v1):  
                self.l.append(v1[i])  
            if i < len(v2):  
                self.l.append(v2[i])  
            i += 1  
        self.index = 0  
  
    def next(self):  
        """ 
        :rtype: int 
        """  
        cur = self.l[self.index]  
        self.index += 1  
        return cur  
  
    def hasNext(self):  
        """ 
        :rtype: bool 
        """  
        if self.index < len(self.l):  
            return True  
        else:  
            return False  
  
# Your ZigzagIterator object will be instantiated and called as such:  
# i, v = ZigzagIterator(v1, v2), []  
# while i.hasNext(): v.append(i.next())  
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
#284. Peeking Iterator
"""
Given an Iterator class interface with methods: next() and hasNext(), design and implement a PeekingIterator that support the peek() operation -- it essentially peek() at the element that will be returned by the next call to next().

Here is an example. Assume that the iterator is initialized to the beginning of the list: [1, 2, 3].

Call next() gets you 1, the first element in the list.

Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.

You call next() the final time and it returns 3, the last element. Calling hasNext() after that should return false.

Hint:

    Think of "looking ahead". You want to cache the next element.
    Is one variable sufficient? Why or why not?
    Test your design with call order of peek() before next() vs next() before peek().

Follow up: How would you extend your design to be generic and work with all types, not just integer?
"""
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    #http://bookshadow.com/weblog/2015/09/21/leetcode-peeking-iterator/
    # peekFlag : peek has been executed, iter has been moved
    # nextElement :  result of peek
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self.peekFlag = False
        self.nextElement = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if not self.peekFlag:
            self.nextElement = self.iter.next()
            self.peekFlag = True
        return self.nextElement

    def next(self):
        """
        :rtype: int
        """
        if not self.peekFlag:
            return self.iter.next()
        else:
            nextElement = self.nextElement
            self.nextElement = None
            self.peekFlag = False
            return nextElement

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.peekFlag or self.iter.hasNext()

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()   
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
#295. Find Median from Data Stream
"""
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.
Examples:

[2,3,4] , the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

    void addNum(int num) - Add a integer number from the data stream to the data structure.
    double findMedian() - Return the median of all elements so far.

For example:

add(1)
add(2)
findMedian() -> 1.5
add(3) 
findMedian() -> 2

"""
from heapq import *
class MedianFinder:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.small = [] #maxheap, all the values are revert
        self.large = [] # python default meanheap
        # size of small N
        # size of large N+ or N
        

    def addNum(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        if len(self.small) == len(self.large):
            heappush(self.large, -heapq.heappushpop(self.small, -num))
        else: #large size already == small size +1
            heappush(self.small, -heapq.heappushpop(self.large, num))
        

    def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        if len(self.small) == len(self.large):
            return float(self.large[0]-self.small[0])/2
        else:
            return float(self.large[0])
        
        

# Your MedianFinder object will be instantiated and called as such:
# mf = MedianFinder()
# mf.addNum(1)
# mf.findMedian()
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
#299. Bulls and Cows
"""
ou are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.

For example:

Secret number:  "1807"
Friend's guess: "7810"

Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)

Write a function to return a hint according to the secret number and friend's guess, use A to indicate the bulls and B to indicate the cows. In the above example, your function should return "1A3B".

Please note that both secret number and friend's guess may contain duplicate digits, for example:

Secret number:  "1123"
Friend's guess: "0111"

In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, and your function should return "1A1B".

You may assume that the secret number and your friend's guess only contain digits, and their lengths are always equal.
"""
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        A = 0
        B = 0
        n = len(secret)
        secret = [int(x) for x in secret]
        guess = [int(x) for x in guess]
        count = [0 for i in xrange(10)] # since only 0-9 ten numbers
        for i in xrange(n):
            count[secret[i]] +=1
            count[guess[i]] -=1
        # count[i] > 0 means for number i, more in secret then in guess
        for i in xrange(n):
            g = guess[i]
            s = secret[i]
            if g == s : A+=1
            else:
                if count[g] > 0: #there are extra g in secret
                    B+=1
                if count[s] <0: #there are extra s in guess
                    B+=1
                count[g] -=1
                count[s]+=1
        return str(A)+'A'+str(B)+'B'
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
#301. Remove Invalid Parentheses
"""
 Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).

Examples:

"()())()" -> ["()()()", "(())()"]
"(a)())()" -> ["(a)()()", "(a())()"]
")(" -> [""]

"""
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        #bfs + pruning
        # http://bookshadow.com/weblog/2015/11/05/leetcode-remove-invalid-parentheses/
        self.visited = set([s]) #!important
        #return self.dfs(s)
        return self.bfs(s)
        
    def dfs(self,s):
        cur = self.cal(s)
        if cur == 0: return [s]
        ans = []

        for i in xrange(len(s)):
            if s[i] in ('(',')'):
                ns = s[:i]+s[i+1:]
                if ns not in self.visited and self.cal(ns) < cur:
                    self.visited.add(ns)
                    ans.extend(self.dfs(ns))
        return ans
        
    def bfs(self,s):
        ans = []
        queue = [s]
        done = False
        while queue!= []:
            t = queue.pop(0)
            cur = self.cal(t)
            if cur == 0: 
                done = True
                ans.append(t)
            if done: continue
            for i in xrange(len(t)):
                 if t[i] in ('(',')'):
                     ns = t[:i]+t[i+1:]
                     if ns not in self.visited and self.cal(ns) < cur:
                        self.visited.add(ns)
                        queue.append(ns)
        return ans
        
    def cal(self,s):
        # a is count of left ( imbalance
        # b is count of right ) imblance
        # return total imbalance
        a = b = 0
        for x in s:
            a += {'(':1,')':-1}.get(x,0)
            b += a <0
            #dbg!!!
            a = max(a,0)
        return a + b
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
#-----------------------------------
#309. Best Time to Buy and Sell Stock with Cooldown
"""
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

    You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

Example:

prices = [1, 2, 3, 0, 2]
maxProfit = 3
transactions = [buy, sell, cooldown, buy, sell]

"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        return self.sol2(prices)
        
    def sol2(self,prices):
        #sells[i] : max profit if not hold stock at day[i]
        #buys[i]: max profix if hold stock
        n = len(prices)
        if n<2 : return 0
        buys = [None]*n
        sells = [None]*n
        sells[0] = 0
        sells[1] = max(0,prices[1]-prices[0])
        buys[0]= -prices[0]
        buys[1] = max(-prices[0],-prices[1])
        for i in xrange(2,n):
            sells[i] = max(sells[i-1],buys[i-1]+prices[i])
            # sells[i-2] here indicate the cool down
            buys[i] = max(sells[i-2]-prices[i],buys[i-1])
        return sells[-1]
        
    
    def sol1(self,prices):
        #auxiliary array
        # sell[i] best accumulate profit at day i
        # buy[i] 
        # http://bookshadow.com/weblog/2015/11/24/leetcode-best-time-to-buy-and-sell-stock-with-cooldown/
        
        
        n = len(prices)
        if n <2: return 0
        buy = [None] *n
        sell = [None] *n
        buy[0] = -prices[0]
        sell[0]=0
        for i in xrange(1,n):
            delta = prices[i]-prices[i-1]
            sell[i] = max(buy[i-1]+prices[i],sell[i-1]+delta)
            buy[i] = max(buy[i-1]-delta,sell[i-2]-prices[i] if i >1 else None)
        return max(sell)
        
#-----------------------------------
#310. Minimum Height Trees
"""
 For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

Example 1:

Given n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3

return [1]

Example 2:

Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5

return [3, 4] 
"""
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        # essentially a BFS solution
        #http://bookshadow.com/weblog/2015/11/26/leetcode-minimum-height-trees/
        children = [set() for x in range(n)]
        #children = [set() for x in range(n)]
        for s, t in edges:
            children[s].add(t)
            children[t].add(s)
        leaves = [x for x in range(n) if len(children[x]) <= 1]
        while n > 2:
            n -= len(leaves)
            newLeaves = []
            for x in leaves:
                for y in children[x]:
                    children[y].remove(x)
                    if len(children[y]) == 1:
                        newLeaves.append(y)
            leaves = newLeaves
        return leaves
        
#-----------------------------------
#312. Burst Balloons
"""
 Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

Find the maximum coins you can collect by bursting the balloons wisely.

Note:
(1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
(2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

Example:

Given [3, 1, 5, 8]

Return 167

    nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
   coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
"""
class Solution(object):
    def maxCoins(self, iNums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # https://leetcode.com/discuss/72216/share-some-analysis-and-explanations
        # we can find that for any balloons left the maxCoins does not depends on the balloons already bursted.
        #dp[l][r] : the max coins can get from range l to r not include boundary l and r
                    # all ballons in between have been bursted
        #dp[l][r] = max(dp[l][r],dp[l][m]+dp[r][m]+ nums[l]*nums[m]*nums[r]) 
        # increase
        nums = [1] + [i for i in iNums if i > 0] + [1]
        n = len(nums)
        dp = [[0]*n for _ in xrange(n)]
    
        for k in xrange(2,n): #k is the distance between left and right
            for l in xrange(0,n-k):
                r = l + k
                for m in xrange(l+1,r):
                   dp[l][r] = max(dp[l][r],dp[l][m]+dp[m][r]+ nums[l]*nums[m]*nums[r])  
        
        return dp[0][n - 1]

#-----------------------------------
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
#324. Wiggle Sort II
"""
 Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....

Example:
(1) Given nums = [1, 5, 1, 1, 6, 4], one possible answer is [1, 4, 1, 5, 1, 6].
(2) Given nums = [1, 3, 2, 2, 3, 1], one possible answer is [2, 3, 1, 3, 1, 2].

Note:
You may assume all input has valid answer.

Follow Up:
Can you do it in O(n) time and/or in-place with O(1) extra space? 
"""
class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """

        # def sol2(self,nums):
        size = len(nums)
        snums = sorted(nums)
        for x in range(1, size, 2) + range(0, size, 2):
            nums[x] = snums.pop()       
"""
解法II O(n)时间复杂度+O(1)空间复杂度解法：

1. 使用O(n)时间复杂度的quickSelect算法，从未经排序的数组nums中选出中位数mid

2. 参照解法I的思路，将nums数组的下标x通过函数idx()从[0, 1, 2, ... , n - 1, n] 映射到 [1, 3, 5, ... , 0, 2, 4, ...]，得到新下标ix

3. 以中位数mid为界，将大于mid的元素排列在ix的较小部分，而将小于mid的元素排列在ix的较大部分。

详见：https://leetcode.com/discuss/77133/o-n-o-1-after-median-virtual-indexing
C++伪代码：

void wiggleSort(vector<int>& nums) {
    int n = nums.size();

    // Find a median.
    auto midptr = nums.begin() + n / 2;
    nth_element(nums.begin(), midptr, nums.end());
    int mid = *midptr;

    // Index-rewiring.
    #define A(i) nums[(1+2*(i)) % (n|1)]

    // 3-way-partition-to-wiggly in O(n) time with O(1) space.
    int i = 0, j = 0, k = n - 1;
    while (j <= k) {
        if (A(j) > mid)
            swap(A(i++), A(j++));
        else if (A(j) < mid)
            swap(A(j), A(k--));
        else
            j++;
    }
}

"""
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
        return self.sol2(preorder)
    def sol2(self,preorder):
        """
        Since this is a preorder serialization, degrees are calculated in a top-down fashion, and, tree is a structure that each node has only one indegree and at most two outdegree.
            Positive degree means there are more indegree than outdegree, which violates the definition.
            every time meet a character, degree ++
            if the character is node, degree -=2
            None node "#" will only +1 degree for balance
            total degree == 0
        """
        A = preorder.split(',')
        n = len(A)
        degree = -1 #root has no indegree, compenstate with -1
        for i in xrange(n):
            degree += 1
            if degree >0 : return False
            if A[i].isdigit():
                degree -=2
        return degree == 0

    def sol1(self,preorder):
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
#334. Increasing Triplet Subsequence
"""
 Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

Formally the function should:

    Return true if there exists i, j, k
    such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false. 

Your algorithm should run in O(n) time complexity and O(1) space complexity. 
"""
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # a is the min element
        # b is the second min close to a
        a,b = None,None
        for i in xrange(len(nums)):
            if a is None or nums[i] <= a:
                a = nums[i]
            elif b is None or nums[i] <=b:
                b = nums[i]
            else:
                return True
        return False
#-----------------------------------
#-----------------------------------
#335. Self Crossing
"""
 You are given an array x of n positive numbers. You start at point (0,0) and moves x[0] metres to the north, then x[1] metres to the west, x[2] metres to the south, x[3] metres to the east and so on. In other words, after each move your direction changes counter-clockwise.

Write a one-pass algorithm with O(1) extra space to determine, if your path crosses itself, or not.

Example 1:

Given x = [2, 1, 1, 2],
┌───┐
│   │
└───┼──>
    │

Return true (self crossing)

Example 2:

Given x = [1, 2, 3, 4],
┌──────┐
│      │
│
│
└────────────>

Return false (not self crossing)

Example 3:

Given x = [1, 1, 1, 1],
┌───┐
│   │
└───┼>

Return true (self crossing)

"""
class Solution(object):
    def isSelfCrossing(self, x):
        """
        :type x: List[int]
        :rtype: bool
        """
        """
        Best solution so far.
            The first if checks if current line crosses the line 3 steps ahead of it
            The second if checks if current line crosses the line 4 steps ahead of it
            The third if checks if current line crosses the line 6 steps ahead of it
                case 3 实际上是螺旋递增转螺旋递减
        /*               i-2
            case 1 : i-1┌─┐
                        └─┼─>i
                         i-3
        
                            i-2
            case 2 : i-1 ┌────┐
                         └─══>┘i-3
                         i  i-4      (i overlapped i-4)
        
            case 3 :    i-4
                       ┌──┐ 
                       │i<┼─┐
                    i-3│ i-5│i-1
                       └────┘
                        i-2
        
        */        
        If none of the above condition is satisfied, there must not be any cross
        
        True means cross
        """
        n = len(x)
        if n <= 3: return False
        for i in xrange(3,n):
            #case1
            if x[i] >=x[i-2] and x[i-1] <=x[i-3] : return True
            elif i >=4 and x[i-1] == x[i-3] and x[i]+x[i-4]>=x[i-2] : return True
            elif i >=5 and x[i-2] >= x[i-4] and x[i]+x[i-4] >= x[i-2] \
                and x[i-1] <= x[i-3] and x[i-1]+x[i-5] >= x[i-3] : return True
        return False
        
#-----------------------------------
#336. Palindrome Pairs 
"""
 Given a list of unique words. Find all pairs of distinct indices (i, j) in the given list, so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.

Example 1:
Given words = ["bat", "tab", "cat"]
Return [[0, 1], [1, 0]]
The palindromes are ["battab", "tabbat"]

Example 2:
Given words = ["abcd", "dcba", "lls", "s", "sssll"]
Return [[0, 1], [1, 0], [3, 2], [2, 4]]
The palindromes are ["dcbaabcd", "abcddcba", "slls", "llssssll"]
"""
class Solution(object):
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        #o(len(words) * len(word)^2)
        #since each word is unique,we can create dict {word:idx}
        tb = {word:idx for idx,word in enumerate(words)}

        def ispal(word):
            n= len(word)
            for i in xrange(n/2):
                if word[i]!=word[n-1-i]: return False
            return True
        
        res = set()
        for i in xrange(len(words)):
            
            if words[i]!= "" and ispal(words[i]) and "" in tb:
                res.add((i,tb[""]))
                res.add((tb[""],i))
            rw = words[i][::-1]
            if rw in tb:
                if i!=tb[rw]: #important
                    res.add((i,tb[rw]))
                    res.add((tb[rw],i))

            for j in xrange(1,len(words[i])):
                left,right = words[i][:j],words[i][j:]
                rvleft,rvright = left[::-1],right[::-1]
                if ispal(right) and rvleft in tb:
                    res.add((i,tb[rvleft]))
                if ispal(left) and rvright in tb:
                    res.add((tb[rvright],i))
                    
        return list(res)
#-----------------------------------
#337. House Robber III
"""
 The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:

     3
    / \
   2   3
    \   \ 
     3   1

Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

Example 2:

     3
    / \
   4   5
  / \   \ 
 1   3   1

Maximum amount of money the thief can rob = 4 + 5 = 9. 
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.sol2(root)
    def sol2(self,root):
        """
        use dict to store visited path
        maxBenifit = max(rob(left) + rob(right), root.val + rob(ll) + rob(lr) + rob(rl) + rob(rr))
        """
        map = {}
        
        def dfs(node,path):
            if node is None: return 0
            if path not in map:
                leftn,rightn = node.left,node.right
                ll = lr = rl = rr = None # important, init
                if leftn: ll,lr = leftn.left,leftn.right
                if rightn: rl,rr = rightn.left,rightn.right
                skipit = dfs(leftn,path+'l') + dfs(rightn,path+'r')
                useit = node.val+ dfs(ll,path+'ll') + dfs(lr,path+'lr') + dfs(rl,path+'rl') + dfs(rr,path+'rr')
                map[path] = max(skipit,useit)
            return map[path]
        
        return dfs(root,"")
        
    def sol1(self,root):
        """
        dfs all the nodes of the tree, each node return two number, int[] num, num[0] is the max value while rob this node, num[1] is max value while not rob this value. Current node return value only depend on its children's value.
        """
        def dfs(node):
            if node == None: return [0,0]
            left = dfs(node.left)
            right = dfs(node.right)
            res = [0,0]
            res[0] = node.val+left[1]+right[1] #rob node
            res[1] = max(left[0],left[1])+max(right[0],right[1]) #skip node and robe its children
            return res
        
        tmp = dfs(root)
        return max(tmp)
#-----------------------------------
#-----------------------------------
#338 . Counting Bits
"""
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

Example:
For num = 5 you should return [0,1,1,2,1,2].

Follow up:

    It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
    Space complexity should be O(n).
    Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.

"""
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        return self.sol3(num)
        
    def sol1(self,num):
        """
        ans[n] = ans[n>>1] + (n&1)
        """
        ans = [0]
        for x in xrange(1,num+1):
            ans.append( ans[x>>1] + (x&1))
        return ans
    def sol2(self,num):
        """
        ans[n] = ans[n - highbits(n)] + 1
        highbits(n)表示只保留n的最高位得到的数字。
        highbits(n) = 1<<int(math.log(x,2))
        
        """
        ans = [0]
        for x in range(1, num + 1):
            ans += ans[x - (1<<int(math.log(x,2)))] + 1,
        return ans   
    def sol3(self,num):
        """
        ans[n] = ans[n & (n - 1)] + 1
        """
        ans = [0]
        for x in range(1, num + 1):
            ans += ans[x & (x - 1)] + 1,
        return ans        
#-----------------------------------
#-----------------------------------
#341. Flatten Nested List Iterator
"""
Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Given the list [[1,1],2,[1,1]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].

Example 2:
Given the list [1,[4,[6]]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6]. 
"""
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = []
        self.list = nestedList


    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop()

    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack or self.list:
            if not self.stack:
                self.stack.append(self.list.pop(0))
            while self.stack and not self.stack[-1].isInteger():
                #need to flat the last item
                top = self.stack.pop().getList()
                for e in top[::-1]: #push in reverse order
                    self.stack.append(e)
            if self.stack and self.stack[-1].isInteger():
                return True
        return False
            
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
#-----------------------------------
#342. Power of Four
"""
 Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

Example:
Given num = 16, return true. Given num = 5, return false.

Follow up: Could you solve it without loops/recursion? 
"""
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # 1. power of 2 n&(n-1) ==0 
        # 2. assuming 32bits then & 0x55555555
        return  (num&(num-1)==0) and (num&0x55555555 > 0)
#-----------------------------------
#343. Integer Break
"""
 Given a positive integer n, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get.

For example, given n = 2, return 1 (2 = 1 + 1); given n = 10, return 36 (10 = 3 + 3 + 4). 
Note: you may assume that n is not less than 2. 
"""
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        #return self.dpsol(n)
        return self.O1sol(n)
        
    def O1sol(self,n):
        """
        n / 3 <= 1 时，分为两个数的乘积，尽量均摊
        n / 3 > 1 时，分为若干个3和2的乘积
        n % 3 == 0 时，分为n个3的乘积
        n % 3 == 1 时，分为n-1个3和两个2的乘积
        n % 3 == 2 时，分为n个3和一个2的乘积
        """
        div = n / 3
        if div <= 1:
            return (n / 2) * (n / 2 + n % 2)
        mod = n % 3
        if mod == 0:
            return 3 ** div
        elif mod == 1:
            return 3 ** (div - 1) * 4
        elif mod == 2:
            return 3 ** div * 2

    def ONsol(self,n):
        return max([self.mulSplitInt(n, m) for m in range(2, n + 1)])
    
    def mulSplitInt(self, n, m):
        quotient = n / m
        remainder = n % m
        return quotient ** (m - remainder) * (quotient + 1) ** remainder
        

    def dpsol(self,n):
        #the element is either 2 or 3
        # so dp[n]=max(3*dp[n-3],2*dp[n-2])
        #special handing for n<=3
        if n<=3: return n-1
        dp= [0 for _ in xrange(n+1)]
        dp[2] = 2
        dp[3]=3
        for i in xrange(4,n+1):
            dp[i] = max(3*dp[i-3],2*dp[i-2])
        return dp[n]
#-----------------------------------
#344. Reverse String
class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]
#-----------------------------------
#345. Reverse Vowels of a String
"""
Write a function that takes a string as input and reverse only the vowels of a string.
Write a function that takes a string as input and reverse only the vowels of a string.

Example 1:
Given s = "hello", return "holle".

Example 2:
Given s = "leetcode", return "leotcede". 

2 pointer method
"""
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        db = ('a', 'e', 'i', 'o', 'u')
        n = len(s)
        l,r = 0,n-1
        ls = list(s)
        while True:
            while l<n and s[l].lower() not in db:
                l+=1
            while r >=0 and s[r].lower() not in db:
                r -=1
            if l>= r: break
            ls[l],ls[r] = ls[r],ls[l]
            l += 1
            r -= 1
        return "".join(ls)
#-----------------------------------
#-----------------------------------
#347. Top K Frequent Elements
"""
 Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].

Note:

    You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
    Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        return self.sol2(nums,k)

    def sol2(self,nums,k):
        #bucket sort o(n)
        n = len(nums)
        freq = {} #key is num, val is freq
        for x in nums:
            freq[x] = freq.get(x,0) +1
        freqList = [[] for _ in xrange(n+1)] #index is freq, value is the number
        for x in freq:
            freqList[freq[x]].append(x)
        ans = []
        for i in xrange(n,0,-1):
            if freqList[i] != []:
                ans += freqList[i]
        return ans[:k]
            
    def sol1(self,nums,k):
        # easy solution is using collections.Counter(nums).most_common(k) method
        # hashtable + heap
        #o(nlogn)
        db = {}
        for x in nums:
            db[x] = db.get(x,0) +1
        import heapq
        #heapq is a minheap
        res = []
        for key in db:
            heapq.heappush(res,[-1*db[key],key])
        ans = []
        for _ in xrange(k):
            ans.append(heapq.heappop(res)[1])
        return ans
#-----------------------------------
#-----------------------------------
#349 Intersection of Two Arrays
"""
 Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

Note:

    Each element in the result must be unique.
    The result can be in any order.

"""
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return list(set(nums1).intersection(set(nums2)))
#-----------------------------------
#350. Intersection of Two Arrays II
"""
 Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].

Note:

    Each element in the result should appear as many times as it shows in both arrays.
    The result can be in any order.

Follow up:

    What if the given array is already sorted? How would you optimize your algorithm?
    What if nums1's size is small compared to nums2's size? Which algorithm is better?
    What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

"""
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return self.sol2(nums1,nums2)
        
    def sol1(self,nums1,nums2):
        """
        sorting + 2pointers since we don't care about the output sequece
        """
        nums1 = sorted(nums1)
        nums2 = sorted(nums2)
        ans = []
        p1 = p2 = 0
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] < nums2[p2]: p1+=1
            elif nums1[p1] > nums2[p2]: p2+=1
            else:
                ans.append(nums1[p1])
                p1 += 1
                p2 +=1
        return ans
    def sol2(self,nums1,nums2):
        """
        the longer one is on disk
        using collections.Counter
        """
        if len(nums1) > len(nums2):
            nums1,nums2 = nums2,nums1
        c= collections.Counter(nums1)
        ans = []
        for x in nums2:
            if c[x] >0:
                ans.append(x)
                c[x] -=1
        return ans
#-----------------------------------
#372. Super Pow
"""
 Your task is to calculate ab mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.

Example1:

a = 2
b = [3]

Result: 8

Example2:

a = 2
b = [1,0]

Result: 1024

"""
class Solution(object):

    def superPow(self, a, b):
        """
        :type a: int
        :type b: List[int]
        :rtype: int
        """
        #2^23 = (2^2)^10 * 2^3,
        # http://www.cnblogs.com/grandyang/p/5651982.html
        res = 1
        for i in xrange(len(b)):
            res = self.pow(res,10)*self.pow(a,b[i])%1337
        return res
        
    def pow(self,x,n):
        if n == 0: return 1
        if n == 1: return x%1337
        return self.pow(x%1337,n/2) *self. pow(x%1337,n-n/2) %1337
#-----------------------------------
#373. Find K Pairs with Smallest Sums 
"""
 You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u,v) which consists of one element from the first array and one element from the second array.

Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums. 
"""
class Solution(object):
    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        #http://bookshadow.com/weblog/2016/07/07/leetcode-find-k-pairs-with-smallest-sums/
        ans = []
        heap = []
        def push(i,j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(heap,[nums1[i]+nums2[j],i,j])
        push(0,0)
        while heap and len(ans) < k:
            _,i,j = heapq.heappop(heap)
            ans.append([nums1[i],nums2[j]])
            push(i,j+1)
            if j ==0:
                push(i+1,0)
        return ans