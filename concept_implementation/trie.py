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
