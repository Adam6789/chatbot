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


def get_sequence_length(df,vocab):
    '''
    Inputs: df, DataFrame
    Outputs: sequence_length, int
                
    '''
    a = df.copy()

    a['words in question'] = a["question"].apply(lambda n: len(vocab.clean_text(n)))
    longest_question = a.sort_values(by=['words in question'],ascending=False).iloc[0,5]

    a['words in answer'] = a["answer"].apply(lambda n: len(vocab.clean_text(n[2:-2])))
    longest_answer = a.sort_values(by=['words in answer'],ascending=False).iloc[0,6]
    
    # the second constant is for the SOS resp. EOS which is in both the question and each answer (input and target)
    # the first constant is for the other token which is additionaly only in the question
    sequence_length = max(longest_question+1, longest_answer)+1
    return sequence_length

def get_answers(df, vocab):
    '''
    Inputs: df, DataFrame
            vocab, Vocabulary object containing the dictionary with all words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
    b = df.copy()
    #b['hello'] = b["question"].apply(lambda n: n.split())
    b['servus'] = b["answer"].apply(lambda n: vocab.clean_text(n[2:-2]))
    answers = b['servus'].tolist()
    new_answers = []
    sequence_length = get_sequence_length(df, vocab)
    for answer in answers:
        new_answer = []
        for word in answer:
            new_answer.append(vocab.words[word])
        padding_size = sequence_length - len(new_answer) - 1
        new_answer += [vocab.words["<EOS>"]] + [vocab.words["<PAD>"]]*padding_size
        new_answers.append(new_answer)
    return torch.LongTensor(new_answers)

def get_questions(df, vocab):
    '''
    Inputs: df, DataFrame
            vocab, Vocabulary object containing the dictionary with all words
    Outputs: new_answers, list with words as numbers followed by an EOS token and potentially one or multiple PAD tokens
    
    '''
    b = df.copy()
    #b = b[:5]
    b['hello'] = b["question"].apply(lambda n: vocab.clean_text(n))
    #b['servus'] = b["answer"].apply(lambda n: n[0].split())
    questions = b['hello'].tolist()
    new_questions = []
    sequence_length = get_sequence_length(df, vocab)
    for question in questions:
        new_question = []
        for word in question:
            new_question.append(vocab.words[word])
        padding_size = sequence_length - len(new_question) - 2
        new_question = [vocab.words["<SOS>"]] + new_question + [vocab.words["<EOS>"]] + [vocab.words["<PAD>"]]*padding_size
        new_questions.append(new_question)
    return torch.LongTensor(new_questions)
        
    