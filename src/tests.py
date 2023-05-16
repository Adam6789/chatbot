import pytest
import pandas as pd
import src.utils
import torch
import random

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
    batches = src.utils.heteroDataLoader(single_samples, batch_size, shuffle)
        
    assert len(batches) == 12, f"The number of batches should be {len_batches}."
    assert torch.all(torch.eq(single_samples[0],batches[0][0])).item() or torch.all(torch.eq(single_samples[1],batches[0][1])).item() or shuffle == False, "Shuffling seems not to work."
    assert len(batches[0]) == batch_size, f"Size of each batch must be {batch_size}."
    
    
    
    
    
    
    