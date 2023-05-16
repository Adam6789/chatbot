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
        
    def getIndex(self, word):
        if self.words.get(word) == None:
            return 3
        else:
            return self.words.get(word)
        
def clean_text(text):
    ps = PorterStemmer()
    text = "".join([ps.stem(c.lower()) for c in text if c not in string.punctuation])
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)    
    return text
            
def get_max_length(sequences):
    '''
    Inputs: sequences, list with sequences
            vocab, vocabulary
    Outputs: sequence_length, int
                
    '''
    sequence_length = max([len(n) for n in sequences])

    return sequence_length

def tokenize_answers(sequences, vocab):
    '''
    Inputs: answers, list
            vocab, dictionary with words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
   
    tokenized_sequences = []
    #max_length = get_max_length(sequences)
    for sequence in sequences:
        tokenized_sequence = []
        for word in sequence:
            tokenized_sequence.append(vocab.getIndex(word))
        #padding_size = max_length - len(tokenized_sequence)
        tokenized_sequence += [vocab.words["<EOS>"]] #+ [vocab.words["<PAD>"]]*padding_size
        tokenized_sequences.append(torch.LongTensor(tokenized_sequence))
    return tokenized_sequences

def tokenize_questions(questions, vocab):
    '''
    Inputs: questions, list
            vocab, dictionary with words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
    new_questions = []
    #max_length = get_max_length(questions)
    for i, question in enumerate(questions):
        new_question = []
        for word in question:
            new_question.append(vocab.getIndex(word))
        #padding_size = max_length - len(new_question)
        new_question = [vocab.words["<SOS>"]] + new_question + [vocab.words["<EOS>"]] #+ [vocab.words["<PAD>"]]*padding_size
        new_questions.append(torch.LongTensor(new_question))
    return new_questions

def tokenize_question(questions, vocab):
    '''
    no vocab increase!!!
    
    Inputs: questions, list
            vocab, dictionary with words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
    new_questions = []
    
    for i, question in enumerate(questions):
        new_question = []
        new_question = [vocab.words["<SOS>"]] + new_question + [vocab.words["<EOS>"]] #+ [vocab.words["<PAD>"]]*padding_size
        new_questions.append(torch.LongTensor(new_question))
    return new_questions

def heteroDataLoader(single_samples, batch_size, shuffle = True):
    
    """
    Inputs:
    -------
    dataset: list
        A list of single samples.
    Outputs:
    --------
    batches: list
        A list of lists with each having multiple samples.
    """
    len_batches = len(single_samples) // batch_size
    random.shuffle(single_samples)
    batches = []
    for i in range(len_batches):
        batches.append(single_samples[i*batch_size:(i+1)*batch_size])
    return batches
        
    