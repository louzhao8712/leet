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
        queue = []
        for letter in word:
            queue.append((letter, node))
            child = node.childs.get(letter)
            if child is None:
                return False
            node = child
        if not node.isWord:
            return False
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