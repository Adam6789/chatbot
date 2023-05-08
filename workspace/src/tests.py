import pytest
import pandas as pd
import src.utils
import torch

@pytest.fixture()
def vocab():
    vocab = utils.Vocab("test")
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    OUT = "<OUT>"
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<OUT>"]
    for word in special_tokens + ["hans", "peter", "ralph"]:
        vocab.indexWord(word)
    return vocab
        
        
@pytest.fixture()
def df():
    df = pd.DataFrame([{ "context":0, "question":"hans peter hans hans", "answer": ["hans"], "answer_start":1},{"context":0,"question":"hans peter", "answer": ["peter","ralph"], "answer_start":1}])
    return df

        
def test_get_answers(df, vocab):
    questions = utils.get_questions(df, vocab)
    should_state = utils.get_sequence_length(df)
    is_state = questions.shape[1]
    print(is_state, should_state)
    assert questions.shape[1] == should_state, f"Each answer should have eventually {should_state} tokens. But there are {is_state} instead."
    assert isinstance(questions, torch.Tensor), "The output should be of Type torch.Tensor."
    
    
    