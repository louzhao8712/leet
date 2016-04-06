class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
	tb={}
	for i in xrange(len(nums)):
		tmp = target = nums[i]
		if tmp in tb:
			return [tb[tmp],i]
		else:
			tb[nums[i]]=i
			
	
