import torch
from nltk.tokenize import RegexpTokenizer

class Vocab:
    def __init__(self, name):
        self.name = name
        self.index = {}
        self.count = 0
        self.words = {}
        
    def clean_text(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        return text
                                    
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
    max_length = get_max_length(sequences)
    for sequence in sequences:
        tokenized_sequence = []
        for word in sequence:
            tokenized_sequence.append(vocab.getIndex(word))
        padding_size = max_length - len(tokenized_sequence)
        tokenized_sequence += [vocab.words["<EOS>"]] + [vocab.words["<PAD>"]]*padding_size
        tokenized_sequences.append(tokenized_sequence)
    return torch.LongTensor(tokenized_sequences)

def tokenize_questions(questions, vocab):
    '''
    Inputs: questions, list
            vocab, dictionary with words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
    new_questions = []
    max_length = get_max_length(questions)
    for i, question in enumerate(questions):
        new_question = []
        for word in question:
            new_question.append(vocab.getIndex(word))
        padding_size = max_length - len(new_question)
        new_question = [vocab.words["<SOS>"]] + new_question + [vocab.words["<EOS>"]] + [vocab.words["<PAD>"]]*padding_size
        new_questions.append(new_question)
    return torch.LongTensor(new_questions)
        
    