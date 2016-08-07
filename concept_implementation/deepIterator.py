from pdb import set_trace as bkp
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el)) #extend
        else:
            result.append(el) #append
    return result
    
def flattenGemerator(x):

    for el in x:
        #if hasattr(el, "__iter__") and not isinstance(el, basestring):
        if hasattr(el, "__iter__") :
            for sub in flatten(el):
                yield sub
        else: #base string
            yield el


L = [[[1, 2, 3], [4, 5]], 6]
#Where the desired output is [1, 2, 3, 4, 5, 6]
print flatten(L)

cc = flattenGemerator( [0,[1,2], 3 ,[4,[5, 6]]])

for i in cc:
    print i