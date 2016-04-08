class Solution(object):
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