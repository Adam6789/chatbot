class Vocab:
    def __init__(self, name):
        self.name = name
        self.index = {}
        self.count = 0
        self.words = {}
                                       
    def indexWord(self, word):
        if word not in self.words:
            self.words[word] = self.count
            self.index[str(self.count)] = word
            self.count += 1
            return True
        else:
            return False
        




    
