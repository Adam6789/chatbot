import torch
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import string
import random

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
        




    
