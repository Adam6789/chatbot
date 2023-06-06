import pytest
import pandas as pd
import torch
import random

from train_chatbot import heteroDataLoader as heteroDataLoader
from models_chatbot import Seq2Seq

@pytest.fixture()
def question():
    question = [torch.LongTensor([1,2,3])]
    return question

@pytest.fixture()
def answer():
    answer = [torch.LongTensor([2,4,6])]
    return answer

@pytest.fixture()
def vocab_source():
    vocab_source = vocab_chatbot.Vocab("Source")
    vocab_source.words = {"A":1,"B":2,"C":3}
    vocab_source.index = {1:"A",2:"B",3:"C"}
    return vocab_source

@pytest.fixture()
def vocab_target():
    vocab_target = vocab_chatbot.Vocab("Target")
    vocab_target.words = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #vocab_target.index = {1:"A",2:"B",3:"C"}
    return vocab_target

def test_seq2seq_forward(question, answer, vocab_source, vocab_target):
    model = Seq2Seq(3, 4, 6, vocab_source, vocab_target, dropout_E=0, dropout_D=0, teacher_forcing_ratio=1,certain=True)
    prediction = model(question, answer)
    assert len(prediction) == len(answer), f"Length of prediction does not match length of answer: {len(prediction)} vs. {len(answer)}"
    return prediction
    

# @pytest.fixture()
# def vocab():
#     vocab = src.utils.Vocab("test")
#     PAD = "<PAD>"
#     SOS = "<SOS>"
#     EOS = "<EOS>"
#     OUT = "<OUT>"
#     special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<OUT>"]
#     for word in special_tokens + ["hans", "peter", "ralph"]:
#         vocab.indexWord(word)
#     return vocab
        
        
# @pytest.fixture()
# def df():
#     df = pd.DataFrame([{ "context":0, "question":"hans peter hans hans", "answer": ["hans"], "answer_start":1},{"context":0,"question":"hans peter", "answer": ["peter","ralph"], "answer_start":1}])
#     return df

        
# def test_get_answers(df, vocab):
#     questions = src.utils.get_questions(df, vocab)
#     should_state = src.utils.get_sequence_length(df)
#     is_state = questions.shape[1]
#     print(is_state, should_state)
#     assert questions.shape[1] == should_state, f"Each answer should have eventually {should_state} tokens. But there are {is_state} instead."
#     assert isinstance(questions, torch.Tensor), "The output should be of Type torch.Tensor."

@pytest.fixture()
def single_samples():
    output = [torch.LongTensor([1,2,3])]*50 + [torch.LongTensor([3,2,1])]*50
    return output

def test_heteroDataLoader(single_samples):
    shuffle = True
    batch_size=8
    batches = heteroDataLoader(single_samples, batch_size, shuffle)
        
    assert len(batches) == 12, f"The number of batches should be {len_batches}."
    assert torch.all(torch.eq(single_samples[0],batches[0][0])).item() or torch.all(torch.eq(single_samples[1],batches[0][1])).item() or shuffle == False, "Shuffling seems not to work."
    assert len(batches[0]) == batch_size, f"Size of each batch must be {batch_size}."
    
    
    
    
    
    
    