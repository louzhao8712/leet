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

#-----------------------------------
class Solution10(object):
#-----------------------------------
class Solution11(object):
#-----------------------------------
class Solution12(object):
#-----------------------------------
class Solution13(object):
#-----------------------------------
class Solution14(object):
#-----------------------------------
class Solution15(object):
#-----------------------------------
class Solution16(object):
#-----------------------------------
class Solution17(object):
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------
#-----------------------------------